import tensorflow as tf
import os
from model import StructuralInpainting
from utils import image_dataset_iterator

img_height = 128
img_width = 128
batch_size = 64
tfrecord_dir = '../data/celeba_tfrecords_128'
vgg16_npy_path = '../data/vgg16.npy'
device = '/gpu:0'
ckpt_dir = './ckpt'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    iterator = image_dataset_iterator(tfrecord_dir, batch_size)
    image_dataset = iterator.get_next()['image']
    "----build graph ----"
    structural_inpainting = StructuralInpainting(batch_size, img_height, img_width, vgg16_npy_path, ckpt_dir)

    sess = tf.Session()
    "get epoch start"
    epoch_start = 0
    if ckpt_dir:
        lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if lasted_checkpoint is not None:
            epoch_start = int(lasted_checkpoint.split('/')[-1].split('-')[-1]) + 1

    for e in range(epoch_start, 60):
        sess.run(iterator.initializer)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                images = sess.run(image_dataset)
                if images.shape[0] != batch_size:
                    print('count of input images is not equal to batch size')
                    break
                if e < 50:
                    structural_inpainting.train_context_encoder(images)
                else:
                    structural_inpainting.train_context_encoder_and_discriminator_jointly(images)
        except tf.errors.OutOfRangeError:
            print('Done training, epoch reached')
        finally:
            coord.request_stop()
            coord.join(threads)

        "save model in the end of epoch"
        structural_inpainting.save_model(e)


if __name__ == '__main__':
    train()