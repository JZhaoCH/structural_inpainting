import tensorflow as tf
import os
from model import StructuralInpainting
from utils import image_dataset_iterator, save_sample

img_height = 128
img_width = 128
batch_size = 64
tfrecord_dir = '../data/celeba_tfrecords_128'
vgg16_npy_path = '../data/vgg16.npy'
device = '/gpu:0'
sample_dir = './samples'
summary_log_dir = './summary'
load_model = True
ckpt_dir = './ckpt'
total_image_count = 202500
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    iterator = image_dataset_iterator(tfrecord_dir, batch_size)
    ground_true = iterator.get_next()['image']
    "----build graph ----"
    input_images = tf.placeholder(tf.float32, (batch_size, img_height, img_width, 3))
    structural_inpainting = StructuralInpainting(batch_size, img_height, img_width, vgg16_npy_path)
    context_encoder_opt, dis_opt, inp_opt, struct_loss_summary, dis_loss_summary, inp_loss_summary, inpainting_images=\
            structural_inpainting.build_graph(input_images)
    "----session config-----"
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=session_config) as sess:
        "----- restore from ckpt file-----"
        saver = tf.train.Saver()
        iteration_num = 0
        if load_model:
            lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(sess, lasted_checkpoint)
                print('load model:', lasted_checkpoint)
                iteration_num = int(lasted_checkpoint.split('/')[-1].split('-')[-1]) + 1
                print('iteration_num is %d' % iteration_num)
            else:
                print('init global variables')
                sess.run(tf.global_variables_initializer())

        "---- training----"
        summary_writer = tf.summary.FileWriter(summary_log_dir, sess.graph)
        epoch_start = iteration_num // (total_image_count//batch_size)
        iter = iteration_num
        for e in range(epoch_start, 50):
            sess.run(iterator.initializer)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    images = sess.run(ground_true)
                    if images.shape[0] != batch_size:
                        print('count of ground true is not equal to batch size')
                        break
                    if e < 50:
                        "-----training without discriminator"
                        if iter % 1000 == 0:
                            _, samples = sess.run([context_encoder_opt, inpainting_images], feed_dict={input_images: images})
                            path = os.path.join(sample_dir, 'epoch_%d_iter_%d.png' % (e, iter))
                            save_sample(samples, [4, 4], path)
                            print('save image:%s' % path)
                        if iter % 100 == 0:
                            _, summary = sess.run([context_encoder_opt, struct_loss_summary], feed_dict={input_images: images})
                            summary_writer.add_summary(summary, global_step=iter)
                        else:
                            sess.run(context_encoder_opt, feed_dict={input_images: images})
                    else:
                        "-----training with discriminator"
                        if iter % 1000 == 0:
                            "----train discriminator----"
                            _, samples = sess.run([dis_opt, inpainting_images], feed_dict={input_images: images})
                            path = os.path.join(sample_dir, 'epoch_%d_iter_%d.png' % (e, iter))
                            save_sample(samples, [4, 4], path)
                            print('save image:%s' % path)
                            "----train context encoder"
                            sess.run(inp_opt, feed_dict={input_images: images})
                        if iter % 100 == 0:
                            "----train discriminator----"
                            _, summary = sess.run([dis_opt, dis_loss_summary], feed_dict={input_images:images})
                            summary_writer.add_summary(summary, global_step=iter)
                            "----train context encoder"
                            _, summary = sess.run([inp_opt, inp_loss_summary], feed_dict={input_images:images})
                            summary_writer.add_summary(summary, global_step=iter)
                        else:
                            sess.run(dis_opt, feed_dict={input_images: images})
                            sess.run(inp_opt, feed_dict={input_images: images})
                    "----save model ---"
                    if iter % 1000 == 0:
                        if not os.path.exists(ckpt_dir):
                            os.mkdir(ckpt_dir)
                        ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
                        saver.save(sess, ckpt_path, global_step=iter)
                        print('save ckpt:', ckpt_path)
                    "-----iter ++ "
                    iter += 1
            except tf.errors.OutOfRangeError:
                print('Done training, epoch reached')
            finally:
                coord.request_stop()
                coord.join(threads)

            "save model in the end of epoch"
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
            saver.save(sess, ckpt_path, global_step=iter)
            print('save ckpt:', ckpt_path)


if __name__ == '__main__':
    train()