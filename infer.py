import numpy as np
import tensorflow.compat.v1 as tf
import argparse

from model import CBSN
from utils import *
import skimage.io as sio

tf.disable_v2_behavior()

def inference(args):
    N = args.num_block
    filters = args.channel
    pad = 32 # pad boundary
    imagefiles = glob.glob(args.imagepath)
    savepath = args.savepath
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    image_test = tf.placeholder(tf.float32, [None, None, None, 3])
    out_test = CBSN(image_test, filters, N, False)
    t_vars = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=t_vars)
        saver.restore(sess,args.modelpath)
        print('Restore latest checkpoint')

        for imgname in imagefiles:
            img = sio.imread(imgname)
            img = (img / 255.).astype(np.float)
            img = np.expand_dims(img, 0)
            img_norm, mean, std = normalize(img)
            img_norm = np.pad(img_norm, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'reflect')
            pred = sess.run(out_test, feed_dict={image_test: img_norm})
            pred = pred[:, pad:-pad, pad:-pad, :]
            pred = std * pred + mean
            pred = im2uint8(pred)
            sio.imsave(os.path.join(savepath,os.path.basename(imgname)),pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parametric")
    parser.add_argument("--num_block", type=int, default=9, help="Number of DCMs")
    parser.add_argument("--channel", type=int, default=128, help="Channel of DBSN")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--modelpath", type=str, default='./ckpt/CBSN_SIDDtrain.ckpt', help='Checkpoint path')
    parser.add_argument("--imagepath", default='./Figures/samplenoisy*.png', help='Glob of input noisy image')
    parser.add_argument("--savepath", default='./results/', help='Output image folder')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    inference(args)
