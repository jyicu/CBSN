import numpy as np
import tensorflow as tf


def conv(x, size, n_out, dilation, act, name):
    out = tf.layers.conv2d(x, n_out, size, 1, padding='same', dilation_rate=dilation,
                           kernel_initializer=tf.variance_scaling_initializer(2.0), name=name, activation=act)
    return out


def masked_conv(x, size, n_out, dilation, mask, act, name):
    with tf.variable_scope(name):
        wshape = [size, size, x.shape[-1].value, n_out]
        wstd = np.sqrt(2) / np.sqrt(np.prod(wshape[:-1]))
        W = tf.get_variable('W1', shape=wshape, initializer=tf.initializers.random_normal(0., wstd))
        b = tf.get_variable('b1', shape=[n_out], initializer=tf.initializers.zeros())

        out = tf.nn.conv2d(x, mask * W, strides=[1] * 4, dilations=[1, dilation, dilation, 1],
                           padding='SAME') + tf.reshape(b, [1, 1, 1, -1])
    return act(out)


def DCM(x, filters, dilation, name):
    with tf.variable_scope(name):
        f = conv(x, 3, filters, dilation, tf.nn.relu, 'conv1')
        f = conv(f, 1, filters, 1, tf.nn.relu, 'conv2')
    return x + f


def make_mask(size):
    mask = np.ones([size, size, 1, 1], np.float32)
    mask[size // 2, size // 2] = 0
    mask = tf.constant(mask)
    return mask


def CBSN(x, filters=128, num_module=9, is_masked=True, reuse=False, name='DBSN'):
    if is_masked:
        mask3 = make_mask(3)
        mask5 = make_mask(5)
    else:
        mask3 = 1
        mask5 = 1
    with tf.variable_scope(name, reuse=reuse):
        conv0 = conv(x, 1, filters, 1, tf.nn.relu, 'conv0')
        B1 = masked_conv(conv0, 3, filters, 1, mask3, tf.nn.relu, 'B1/conv1')
        B1 = conv(B1, 1, filters, 1, tf.nn.relu, 'B1/conv2')
        B1 = conv(B1, 1, filters, 1, tf.nn.relu, 'B1/conv3')
        B2 = masked_conv(conv0, 5, filters, 1, mask5, tf.nn.relu, 'B2/conv1')
        B2 = conv(B2, 1, filters, 1, tf.nn.relu, 'B2/conv2')
        B2 = conv(B2, 1, filters, 1, tf.nn.relu, 'B2/conv3')
        for i in range(num_module):
            B1 = DCM(B1, filters, 2, 'B1/DCM' + repr(i))
            B2 = DCM(B2, filters, 3, 'B2/DCM' + repr(i))
        B1 = conv(B1, 1, filters, 1, tf.nn.relu, 'B1/conv4')
        B2 = conv(B2, 1, filters, 1, tf.nn.relu, 'B2/conv4')
        concat = tf.concat([B1, B2], axis=-1)
        f = conv(concat, 1, filters, 1, tf.nn.relu, 'conv1')
        f = conv(f, 1, 64, 1, tf.nn.relu, 'conv2')
        f = conv(f, 1, 64, 1, tf.nn.relu, 'conv3')
        out = conv(f, 1, 3, 1, None, 'conv4')
        return out
