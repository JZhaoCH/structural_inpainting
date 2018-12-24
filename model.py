import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from vgg16 import Vgg16Feature
import os


class StructuralInpainting:
    def __init__(self, batch_size, img_height, img_width, vgg16_npy_path, ckpt_dir, predicting_mode=False):
        self._batch_size = batch_size
        self._img_height = img_height
        self._img_width = img_width
        self._img_channel = 3
        self._missing_area_size = (56, 56)
        self._ce_output_shape = (64, 64, 3)
        self._mask = self._create_mask()
        self._overlapping_mask = self._create_overlapping_mask()
        self._vgg16 = Vgg16Feature(vgg16_npy_path=vgg16_npy_path)
        self._gray_image_value = 0.437  # default value to set missing area
        self._learning_rate_ce = 2e-4
        self._learning_rate_dis = 2e-5
        self._lambda_adv = 0.1
        self._ckpt_dir = ckpt_dir
        self._summary_log_dir = './summary'
        self._model_name = "model.ckpt"
        self._predicting_mode = predicting_mode
        self._sess = tf.Session()

        if self._predicting_mode is False:
            # training mode
            self._input_image_ph = tf.placeholder(tf.float32, (self._batch_size, self._img_height, self._img_width, self._img_channel))
            self._build_graph_for_training()
            if not os.path.exists(self._summary_log_dir):
                os.mkdir(self._summary_log_dir)
            self.summary_writer = tf.summary.FileWriter(self._summary_log_dir, self._sess.graph)
        else:
            # predicting mode
            self._input_image_ph = tf.placeholder(tf.float32, (self._batch_size, self._img_height, self._img_width, self._img_channel))
            self._build_graph_for_predicting()

        # try to load trained model
        if self._ckpt_dir and os.path.exists(self._ckpt_dir):
            saver = tf.train.Saver()
            lasted_checkpoint = tf.train.latest_checkpoint(self._ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(self._sess, lasted_checkpoint)
                print('load model:', lasted_checkpoint)
            else:
                if self._predicting_mode:
                    raise Exception('predicting mode: can not find trained model in ckpt_dir')
                else:
                    print('init global variables')
                    self._sess.run(tf.global_variables_initializer())
        else:
            if self._predicting_mode:
                raise Exception('predicting mode: ckpt_dir should not be None or ckpt_dir does not exist')
            else:
                print('init global variables')
                self._sess.run(tf.global_variables_initializer())

    def _context_encoder(self, input):
        """
        construct context encoder network
        :param input:
        :return:
        """
        with tf.variable_scope("context_encoder"):
            "----encoder"
            result = layers.conv2d(input, num_outputs=64, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=64, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=128, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=256, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=512, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            "bottleneck"
            result = layers.conv2d(result, num_outputs=2000, kernel_size=4, stride=1, padding="VALID", activation_fn=tf.nn.relu)
            "decoder"
            result = tf.image.resize_images(result, size=[4, 4], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            result = layers.conv2d(result, num_outputs=512, kernel_size=4, stride=1, padding="SAME", activation_fn=tf.nn.relu)
            result = tf.image.resize_images(result, size=[8, 8], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            result = layers.conv2d(result, num_outputs=256, kernel_size=4, stride=1, padding="SAME", activation_fn=tf.nn.relu)
            result = tf.image.resize_images(result, size=[16, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            result = layers.conv2d(result, num_outputs=128, kernel_size=4, stride=1, padding="SAME", activation_fn=tf.nn.relu)
            result = tf.image.resize_images(result, size=[32, 32], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            result = layers.conv2d(result, num_outputs=64, kernel_size=4, stride=1, padding="SAME", activation_fn=tf.nn.relu)
            "output-layer"
            result = tf.image.resize_images(result, size=[64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            result = layers.conv2d(result, num_outputs=3, kernel_size=4, stride=1, padding="SAME")
        return result

    def _discriminator_network(self, input, reuse=False):
        """
        construct discriminator network
        :param input:
        :param reuse:
        :return:
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            result = layers.conv2d(input, num_outputs=32, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=128, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=256, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.flatten(result)
            result = layers.fully_connected(result, num_outputs=1)
            return result

    def _struct_loss(self, ground_true, inpainting):
        """
        create struct loss for training context encoder
        :param ground_true:
        :param inpainting:
        :return:
        """
        recon_loss_ori = tf.square(ground_true-inpainting)
        recon_loss_center = tf.reduce_mean(recon_loss_ori*(1-self._overlapping_mask), axis=[0, 1, 2, 3])  # Loss for non-overlapping region
        recon_loss_overlap = tf.reduce_mean(recon_loss_ori*self._overlapping_mask*10, axis=[0, 1, 2, 3])
        recon_loss = recon_loss_overlap + recon_loss_center
        feature_loss = self._vgg16.feature_loss(ground_true, inpainting)
        struct_loss = recon_loss + feature_loss
        return struct_loss

    def _discriminator_loss(self, ground_logits, inpainting_logits):
        """
        create discriminator loss for training discriminator network
        :param ground_logits:
        :param inpainting_logits:
        :return:
        """
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(ground_logits), logits=ground_logits))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(inpainting_logits), logits=inpainting_logits))
        loss = loss_real + loss_fake
        return self._lambda_adv * loss

    def _inpainting_loss(self, ground_true, inpainting, inpainting_logits):
        """
        create inpainting loss for training context encoder
        :param ground_true:
        :param inpainting:
        :param inpainting_logits:
        :return:
        """
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(inpainting_logits), logits=inpainting_logits))
        struct_loss = self._struct_loss(ground_true, inpainting)
        loss = struct_loss + self._lambda_adv * loss_fake
        return loss

    def _mask_images(self, images):
        """
        mask images
        :param images:
        :return:
        """
        result = tf.multiply(images, self._mask) + (1-self._mask)*self._gray_image_value
        return result

    def _get_central_part_of_images(self, images):
        """
        get the central part of ground truth as input of discriminator
        :param images:
        :return:
        """
        h_start = (self._img_height-self._ce_output_shape[0])//2
        w_start = (self._img_width-self._ce_output_shape[1])//2
        central = images[:, h_start:h_start+self._ce_output_shape[0], w_start:w_start+self._ce_output_shape[1], :]
        return central

    def _create_mask(self):
        """
        create mask to mask the central part of images as input of context encoder
        :return:
        """
        mask = np.ones((self._batch_size, self._img_height, self._img_width, self._img_channel), dtype=np.float32)
        h_start = (self._img_height - self._missing_area_size[0]) // 2
        w_start = (self._img_width - self._missing_area_size[1]) // 2
        mask[:, h_start: h_start + self._missing_area_size[0], w_start: w_start + self._missing_area_size[1], :] = 0
        return mask

    def _create_overlapping_mask(self):
        """
        create the overlapping mask
        :return:
        """
        mask = np.ones((self._batch_size, self._ce_output_shape[0], self._ce_output_shape[1], self._ce_output_shape[2]),
                       dtype=np.float32)
        h_start = (self._ce_output_shape[0] - self._missing_area_size[0]) // 2
        w_start = (self._ce_output_shape[1] - self._missing_area_size[1]) // 2
        mask[:, h_start:h_start+self._missing_area_size[0], w_start:w_start+self._missing_area_size[1], :] = 0
        return mask

    def _combine_masked_images_and_inpainting_area(self, masked_images, inpainting_area):
        """
        combine the masked images and inpainting area from context encoder to get inpainting image
        :param masked_images:
        :param inpainting_area:
        :return:
        """
        h_start = (self._img_height - self._ce_output_shape[0]) // 2
        w_start = (self._img_width - self._ce_output_shape[1]) // 2
        channel = inpainting_area.shape[3]
        "----------------"
        array1 = np.zeros((self._batch_size, h_start, self._ce_output_shape[1], channel), dtype=np.float32)
        array2 = np.zeros((self._batch_size, self._img_height-h_start-self._ce_output_shape[0],
                           self._ce_output_shape[1], channel), dtype=np.float32)
        inpainting_area = tf.concat((array1, inpainting_area, array2), axis=1)
        "-----------------"
        array1 = np.zeros((self._batch_size, self._img_width, w_start, channel), dtype=np.float32)
        array2 = np.zeros((self._batch_size, self._img_width, self._img_width-self._ce_output_shape[1]-w_start, channel), dtype=np.float32)
        inpainting_area = tf.concat((array1, inpainting_area, array2), axis=2)

        inpainting_images = tf.multiply(masked_images, self._mask) + (1-self._mask)*inpainting_area
        return inpainting_images

    def _build_graph_for_training(self):
        """
        build graph, create necessary flow for training
        :return:
        """
        masked_images = self._mask_images(self._input_image_ph)
        inpainting_area = self._context_encoder(masked_images)
        self._inpainting_images = self._combine_masked_images_and_inpainting_area(masked_images, inpainting_area)
        "---"
        ground_true = self._get_central_part_of_images(self._input_image_ph)
        struct_loss = self._struct_loss(ground_true, inpainting_area)

        ground_true_logits = self._discriminator_network(ground_true)
        inpainting_logits = self._discriminator_network(inpainting_area, reuse=True)
        dis_loss = self._discriminator_loss(ground_true_logits, inpainting_logits)
        inp_loss = self._inpainting_loss(ground_true, inpainting_area, inpainting_logits)

        "--------Optimizer"
        ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='context_encoder')
        dis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        self._global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        self._context_encoder_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate_ce, beta1=0.5). \
            minimize(struct_loss, var_list=ce_params, global_step=self._global_step)
        self._dis_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate_dis, beta1=0.5). \
            minimize(dis_loss, var_list=dis_params, global_step=self._global_step)
        self._inp_opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate_dis, beta1=0.5). \
            minimize(inp_loss, var_list=ce_params)

        '-----------summary'
        self._struct_loss_summary = tf.summary.scalar("struct_loss", struct_loss)
        self._dis_loss_summary = tf.summary.scalar("discriminator_loss", dis_loss)
        self._inp_loss_summary = tf.summary.scalar("inpainting_loss", inp_loss)
        self._inpainting_image_summary = tf.summary.image("inpainting_image", self._inpainting_images)

    def _build_graph_for_predicting(self):
        """
        build necessary flow for inpainting by trained model
        :return:
        """
        masked_images = self._mask_images(self._input_image_ph)
        inpainting_area = self._context_encoder(masked_images)
        self._inpainting_images = self._combine_masked_images_and_inpainting_area(masked_images, inpainting_area)

    def train_context_encoder(self, images):
        """
        train context encoder
        :param images:
        :return:
        """
        feed_dict = {self._input_image_ph: images}
        global_step = self._global_step.eval(self._sess)
        if global_step % 1000 == 0:
            _, samples = self._sess.run([self._context_encoder_opt, self._inpainting_image_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(samples, global_step=global_step)
        if global_step % 100 == 0:
            _, summary = self._sess.run([self._context_encoder_opt, self._struct_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        else:
            self._sess.run(self._context_encoder_opt, feed_dict=feed_dict)

    def train_context_encoder_and_discriminator_jointly(self, images):
        """
        train context encoder and discriminator jointly
        :param images:
        :return:
        """
        global_step = self._global_step.eval(self._sess)
        feed_dict = {self._input_image_ph: images}
        if global_step % 1000 == 0:
            "----train discriminator----"
            _, samples = self._sess.run([self._dis_opt, self._inpainting_image_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(samples, global_step=global_step)
            "----train context encoder"
            self._sess.run(self._inp_opt, feed_dict=feed_dict)
        if global_step % 100 == 0:
            "----train discriminator----"
            _, summary = self._sess.run([self._dis_opt, self._dis_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
            "----train context encoder"
            _, summary = self._sess.run([self._inp_opt, self._inp_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        else:
            self._sess.run(self._dis_opt, feed_dict=feed_dict)
            self._sess.run(self._inp_opt, feed_dict=feed_dict)

    def save_model(self, epoch):
        """
        save trained model
        :param epoch:
        :return:
        """
        saver = tf.train.Saver()
        if not os.path.exists(self._ckpt_dir):
            os.mkdir(self._ckpt_dir)
        ckpt_path = os.path.join(self._ckpt_dir, self._model_name)
        saver.save(self._sess, ckpt_path, global_step=epoch)
        print('save ckpt:%s\n' % ckpt_path)

    def complete_image(self, images):
        """
        complete image by using trained model
        :param images:
        :return:
        """
        feed_dict = {self._input_image_ph: images}
        result = self._sess.run(self._inpainting_images, feed_dict=feed_dict)
        return result