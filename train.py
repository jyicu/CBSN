import numpy as np
import tensorflow.compat.v1 as tf
import argparse

from model import CBSN
from utils import *
from dataprocess import *

DATA_PATH = './dataset/'
tf.disable_v2_behavior()

def trainer(args):
    batchsize = args.batchsize
    max_iter = args.maxiter
    ckpt_dir = './ckpts/'
    checkpoint_dir = ckpt_dir + args.name
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with open(checkpoint_dir + '/args.txt', 'w') as f:
        f.write(str(args._get_kwargs()))
    cpy_code(checkpoint_dir)

    image_train = load_tfrecords(DATA_PATH + 'SIDD256.tfrecords',
                                 DATA_PATH + 'DND256.tfrecords',
                                 args.dataset, args.patchsize, 800, batchsize)

    val_set = np.load(DATA_PATH + 'val_SIDD_noisy.npy').astype(np.float32)
    val_gt = np.load(DATA_PATH + 'val_SIDD_gt.npy').astype(np.float32)
    val_num = len(val_set) // batchsize

    image_test = tf.placeholder(tf.float32, [None, None, None, 3])

    N = args.block
    filters = args.channel

    down_ratio1 = tf.constant(args.stride_b)
    down_ratio2 = args.stride_i

    image_ds1 = S2B(image_train, down_ratio1)
    rand_mask = generate_mask(image_train, down_ratio2)
    image_ds2 = make_subimage_from_mask(image_train, rand_mask, down_ratio2)

    step = tf.train.create_global_step()

    out_train_masked1 = CBSN(image_ds1, filters, N, True)
    out_train_masked1 = B2S(out_train_masked1, down_ratio1)
    out_train_masked2 = CBSN(image_ds2, filters, N, True, reuse=True)
    out_train_unmasked = CBSN(image_train, filters, N, False, reuse=True)
    out_train_unmasked_ds = make_subimage_from_mask(out_train_unmasked, rand_mask, down_ratio2)

    out_test = CBSN(image_test, filters, N, False, reuse=True)

    loss_masked_self = tf.reduce_mean(tf.abs(out_train_masked1 - image_train))
    loss_reg = tf.reduce_mean(tf.abs(out_train_unmasked_ds - tf.stop_gradient(out_train_masked2)))
    loss_unmasked_self = tf.reduce_mean(tf.abs(out_train_unmasked - image_train))

    schedule = step / 200000
    schedule = tf.cast(tf.clip_by_value(schedule, 0.0, 1.0), tf.float32)
    total_loss = schedule * (loss_unmasked_self + args.lmbda * loss_reg) + loss_masked_self

    lr = tf.train.exponential_decay(args.lr, step, 100000, 0.5, staircase=True)
    train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, global_step=step)

    get_paramsnum()

    # Summaries for train and validation
    train_summary = tf.summary.merge([tf.summary.scalar("Train/total_loss", total_loss),
                                      tf.summary.scalar("Train/masked_self_loss", loss_masked_self),
                                      tf.summary.scalar("Train/regularization_loss", loss_reg),
                                      tf.summary.scalar("Train/unmasked_self_loss", loss_unmasked_self),
                                      tf.summary.scalar("Train/lr", lr),
                                      ])
    train_image_summary = tf.summary.merge([tf.summary.image("Train_out", out_train_unmasked[0:1])
                                            ])
    val_summary_in = tf.placeholder(tf.float32, [])
    val_summary_img = tf.placeholder(tf.float32, [1, None, None, 3])
    val_summary = tf.summary.merge([tf.summary.scalar("Val/psnr", val_summary_in),
                                    ])
    val_image_summary = tf.summary.merge([tf.summary.image("Denoised", val_summary_img)])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=20)
        latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        if not latest == None:
            saver.restore(sess, save_path=latest)
            print('Restore latest checkpoint')
        start = sess.run(step)
        summary_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        for i in range(start, max_iter):
            if i % args.print_every == 0:
                loss_out, summary, summary_img = sess.run([total_loss, train_summary, train_image_summary])
                print('D: Step {} loss : {}'.format(i, loss_out))
                summary_writer.add_summary(summary, i)
                summary_writer.add_summary(summary_img, i)
            sess.run(train_op)

            # Validation
            if i % args.val_every == 0:
                psnr_val = 0.0
                pred = None
                for j in range(val_num):
                    img = val_set[batchsize * j:batchsize * (j + 1)]
                    gt_img = val_gt[batchsize * j:batchsize * (j + 1)]
                    mean = np.mean(img, axis=(1, 2), keepdims=True)
                    std = np.std(img, axis=(1, 2), keepdims=True)
                    std = np.maximum(std, 1.0 / 256) # 256 x 256 patch
                    img_norm = (img - mean) / std
                    pred = sess.run(out_test, feed_dict={image_test: img_norm})
                    pred = std * pred + mean
                    psnr_val += batch_PSNR_255(pred, gt_img)
                    if j == 0:
                        summary_img = sess.run(val_image_summary,
                                               feed_dict={val_summary_img: (255 * pred[0:1]).astype(np.uint8)})
                        summary_writer.add_summary(summary_img, i)
                psnr_val = psnr_val / val_num
                print('Validation PSNR : {}'.format(psnr_val))
                summary_val = sess.run(val_summary, feed_dict={val_summary_in: psnr_val})
                summary_writer.add_summary(summary_val, i)
            if i % args.save_every == 0:
                saver.save(sess, checkpoint_dir + '/CBSN_iter' + str(i) + '.ckpt')
        saver.save(sess, checkpoint_dir + '/CBSNlast.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parametric")
    parser.add_argument("--batchsize", type=int, default=4, help="Training batch size")
    parser.add_argument("--channel", type=int, default=128, help="Channel of CBSN")
    parser.add_argument("--block", type=int, default=9, help="Blocks of CBSN")
    parser.add_argument("--maxiter", type=int, default=500000, help="Number of training iterations")
    parser.add_argument("--print_every", type=int, default=1000, help='Print interval')
    parser.add_argument("--val_every", type=int, default=10000, help='Validation interval')
    parser.add_argument("--save_every", type=int, default=10000, help='Save interval')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--lmbda", type=float, default=2.0, help="Weight of inv loss")
    parser.add_argument("--stride_b", type=int, default=5, help="Stride in blind loss")
    parser.add_argument("--stride_i", type=int, default=2, help="Stride in downinv loss")
    parser.add_argument("--name", type=str, default='temp', help='Name of experiments')
    parser.add_argument("--dataset", type=str, default='SIDD', help='Training dataset : SIDD or DND')
    parser.add_argument("--patchsize", type=int, default=120, help='Training patch size')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    trainer(args)
