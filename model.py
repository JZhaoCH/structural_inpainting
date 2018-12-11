import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from vgg16 import Vgg16Feature


class StructuralInpainting:
    def __init__(self, batch_size, img_height, img_width, vgg16_npy_path):
        self.batch_size = batch_size
        self.image_shape = (img_height, img_width, 3)
        self.missing_area_size = (56, 56)
        self.ce_output_shape = (64, 64, 3)
        self.mask = self._create_mask()
        self.overlapping_mask = self._create_overlapping_mask()
        self.vgg16 = Vgg16Feature(vgg16_npy_path=vgg16_npy_path)
        self.gray_image_value = 0.437
        self.learning_rate_ce = 2e-4
        self.learning_rate_dis = 2e-5
        self.lambda_adv = 0.01

    def _context_encoder(self, input):
        "input size: 128*128*3"
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
        "input size: 64*64*3"
        with tf.variable_scope('discriminator', reuse=reuse):
            result = layers.conv2d(input, num_outputs=32, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=128, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.conv2d(result, num_outputs=256, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            result = layers.flatten(result)
            result = layers.fully_connected(result, num_outputs=1)
            return result

    def _struct_loss(self, ground_true, inpainting):
        recon_loss_ori = tf.square(ground_true-inpainting)
        recon_loss_center = tf.reduce_mean(tf.sqrt(1e-5 + tf.reduce_sum(recon_loss_ori*(1-self.overlapping_mask), [1, 2, 3]))) / 10.  # Loss for non-overlapping region
        recon_loss_overlap = tf.reduce_mean(tf.sqrt(1e-5 + tf.reduce_sum(recon_loss_ori * self.overlapping_mask, [1, 2, 3])))
        recon_loss = recon_loss_overlap + recon_loss_center
        feature_loss = self.vgg16.feature_loss(ground_true, inpainting)
        struct_loss = recon_loss + feature_loss
        return struct_loss

    def _discriminator_loss(self, ground_logits, inpainting_logits):
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(ground_logits), logits=ground_logits))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(inpainting_logits), logits=inpainting_logits))
        loss = loss_real + loss_fake
        return self.lambda_adv * loss

    def _inpainting_loss(self, ground_true, inpainting, inpainting_logits):
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(inpainting_logits), logits=inpainting_logits))
        struct_loss = self._struct_loss(ground_true, inpainting)
        loss = struct_loss + self.lambda_adv * loss_fake
        return loss

    def _mask_images(self, images):
        result = tf.multiply(images, self.mask) + (1-self.mask)*self.gray_image_value
        return result

    def _get_central_of_images(self, images):
        h_start = (self.image_shape[0]-self.ce_output_shape[0])//2
        w_start = (self.image_shape[1]-self.ce_output_shape[1])//2
        central = images[:, h_start:h_start+self.ce_output_shape[0], w_start:w_start+self.ce_output_shape[1], :]
        return central

    def _create_mask(self):
        mask = np.ones((self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]), dtype=np.float32)
        h_start = (self.image_shape[0] - self.missing_area_size[0]) // 2
        w_start = (self.image_shape[1] - self.missing_area_size[1]) // 2
        mask[:, h_start: h_start + self.missing_area_size[0], w_start: w_start + self.missing_area_size[1], :] = 0
        return mask

    def _create_overlapping_mask(self):
        mask = np.ones((self.batch_size, self.ce_output_shape[0], self.ce_output_shape[1], self.ce_output_shape[2]),
                       dtype=np.float32)
        h_start = (self.ce_output_shape[0] - self.missing_area_size[0]) // 2
        w_start = (self.ce_output_shape[1] - self.missing_area_size[1]) // 2
        mask[:, h_start:h_start+self.missing_area_size[0], w_start:w_start+self.missing_area_size[1], :] = 0
        return mask

    def _get_inpainting_images(self, masked_images, inpainting_area):
        h_start = (self.image_shape[0] - self.ce_output_shape[0]) // 2
        w_start = (self.image_shape[1] - self.ce_output_shape[1]) // 2
        channel = inpainting_area.shape[3]
        "----------------"
        array1 = np.zeros((self.batch_size, h_start, self.ce_output_shape[1], channel), dtype=np.float32)
        array2 = np.zeros((self.batch_size, self.image_shape[0]-h_start-self.ce_output_shape[0],
                           self.ce_output_shape[1], channel), dtype=np.float32)
        inpainting_area = tf.concat((array1, inpainting_area, array2), axis=1)
        "-----------------"
        array1 = np.zeros((self.batch_size, self.image_shape[1], w_start, channel), dtype=np.float32)
        array2 = np.zeros((self.batch_size, self.image_shape[1], self.image_shape[1]-self.ce_output_shape[1]-w_start, channel), dtype=np.float32)
        inpainting_area = tf.concat((array1, inpainting_area, array2), axis=2)

        inpainting_images = tf.multiply(masked_images, self.mask) + (1-self.mask)*inpainting_area
        return inpainting_images

    def build_graph(self, input_images):
        masked_images = self._mask_images(input_images)
        inpainting_area = self._context_encoder(masked_images)
        inpainting_images = self._get_inpainting_images(masked_images, inpainting_area)
        "---"
        ground_true = self._get_central_of_images(input_images)
        struct_loss = self._struct_loss(ground_true, inpainting_area)

        ground_true_logits = self._discriminator_network(ground_true)
        inpainting_logits = self._discriminator_network(inpainting_area, reuse=True)
        dis_loss = self._discriminator_loss(ground_true_logits, inpainting_logits)
        inp_loss = self._inpainting_loss(ground_true, inpainting_area, inpainting_logits)

        "--------Optimizer"
        ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='context_encoder')
        dis_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        context_encoder_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        dis_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        inp_counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        context_encoder_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ce, beta1=0.5). \
            minimize(struct_loss, var_list=ce_params, global_step=context_encoder_counter)
        dis_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_dis, beta1=0.5). \
            minimize(dis_loss, var_list=dis_params, global_step=dis_counter)
        inp_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_dis, beta1=0.5). \
            minimize(inp_loss, var_list=ce_params, global_step=inp_counter)

        '-----------summary'
        struct_loss_summary = tf.summary.scalar("struct_loss", struct_loss)
        dis_loss_summary = tf.summary.scalar("discriminator_loss", dis_loss)
        inp_loss_summary = tf.summary.scalar("inpainting_loss", inp_loss)
        return context_encoder_opt, dis_opt, inp_opt, struct_loss_summary, dis_loss_summary, inp_loss_summary, \
               inpainting_images
