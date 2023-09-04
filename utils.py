import tensorflow.compat.v1 as tf
import glob
import os
import numpy as np
import skimage.measure
from shutil import copy


def get_paramsnum():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('=' * 20)
    print('Network parameters : {}'.format(total_parameters))
    print('=' * 20)
    return


def cpy_code(checkpoint_dir):
    files = glob.glob('./*.py')
    for file in files:
        copy(file, os.path.join(checkpoint_dir, file))


def batch_PSNR_255(noisy, ref):
    PSNR = 0.0
    for i in range(noisy.shape[0]):
        ref_i = np.round(255 * ref[i, :, :, :]).astype(np.uint8)
        noisy_i = np.round(255 * noisy[i, :, :, :]).astype(np.uint8)
        PSNR += skimage.metrics.peak_signal_noise_ratio(ref_i, noisy_i, data_range=255)
    return (PSNR / noisy.shape[0])


def to_valid_image(img):
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.cast(255 * img, tf.uint8)
    return img

def normalize(img):
    mean = np.mean(img, axis=(1, 2), keepdims=True)
    std = np.std(img, axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1 / np.sqrt(img.shape[1] * img.shape[2]))
    img_norm = (img - mean) / std
    return img_norm, mean, std

def im2uint8(img):
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return img.squeeze()
