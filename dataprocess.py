import tensorflow.compat.v1 as tf
import numpy as np
import glob


def tf_normalize(image, patchsize):
    mean = tf.reduce_mean(image, [0, 1], keepdims=True)
    std = tf.math.reduce_std(image, [0, 1], keepdims=True)
    # adjusted_stddev  = tf.maximum(std, 1.0 / np.sqrt(N))
    adjusted_stddev = tf.maximum(std, 1.0 / patchsize)
    normalized = (image - mean) / adjusted_stddev
    return normalized


def _parse_function(example_proto, patch_size=120):
    keys_to_features = {'Noisy': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    noisy = parsed_features['Noisy']
    noisy = tf.divide(tf.cast(tf.decode_raw(noisy, tf.uint8), tf.float32), 255.)
    noisy = tf.reshape(noisy, [256, 256, 3])

    noisy = tf.image.random_crop(noisy, [patch_size, patch_size, 3])
    idx = tf.random.uniform([], 0, 8, tf.int32)
    noisy = augment(noisy, idx)
    noisy = tf_normalize(noisy, patch_size)

    return noisy


def augment(image, seed):
    rot_time = seed % 4
    image = tf.image.rot90(image, rot_time)
    do_flip = tf.cast(seed // 4, tf.bool)
    image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
    return image


def _parse_float(example_proto, patch_size=120):
    keys_to_features = {'Noisy': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    noisy = parsed_features['Noisy']
    noisy = tf.decode_raw(noisy, tf.float32)
    noisy = tf.reshape(noisy, [256, 256, 3])
    noisy = tf.image.random_crop(noisy, [patch_size, patch_size, 3])
    idx = tf.random.uniform([], 0, 8, tf.int32)
    noisy = augment(noisy, idx)
    noisy = tf_normalize(noisy, patch_size)

    return noisy


def load_tfrecords(tfrecords_file, tfrecords_file_DND, mode, patchsize=120, n_shuffle=1000, batch_size=64):
    if mode == 'SIDD':
        dataset = tf.data.TFRecordDataset(tfrecords_file)
        dataset = dataset.map(lambda x: _parse_function(x, patchsize))
    elif mode == 'DND':
        dataset = tf.data.TFRecordDataset(tfrecords_file_DND)
        dataset = dataset.map(lambda x: _parse_float(x, patchsize))
    else:
        raise ValueError("Dataset must be one of 'SIDD' or 'DND'")

    dataset = dataset.shuffle(n_shuffle)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    return x



def generate_mask(img_in, ds_ratio=2):
    d = ds_ratio * ds_ratio
    mask_shape = [tf.size(img_in[:, :, :, 0]) // d]
    random_mask = tf.random_uniform(mask_shape, 0, d, dtype=tf.int32)
    return random_mask


def make_subimage_from_mask(img_in, mask, ds_ratio=2, inputchannel=3):
    img = tf.space_to_depth(img_in, ds_ratio)
    img_shape = tf.shape(img)
    img_flatten = tf.reshape(img, [-1, ds_ratio * ds_ratio, inputchannel])
    gather_idx1 = tf.stack([tf.range(tf.shape(img_flatten)[0]), mask], axis=-1)
    subimage = tf.reshape(tf.gather_nd(img_flatten, gather_idx1), tf.concat([img_shape[:-1], [inputchannel]], 0))
    return subimage


def PD(x, s):
    x_shape = tf.shape(x)
    y = tf.space_to_depth(x, s)
    y = tf.reshape(y, [x_shape[0], x_shape[1] // s, x_shape[2] // s, s, s, x_shape[3]])
    t = tf.transpose(y, [0, 3, 1, 4, 2, 5])
    z = tf.reshape(t, x_shape)
    return z


def PD_inv(x, s):
    x_shape = tf.shape(x)
    y = tf.reshape(x, [x_shape[0], s, x_shape[1] // s, s, x_shape[2] // s, x_shape[3]])
    t = tf.transpose(y, [0, 2, 4, 1, 3, 5])
    z = tf.reshape(t, [x_shape[0], x_shape[1] // s, x_shape[2] // s, x_shape[3] * s * s])
    z = tf.depth_to_space(z, s)
    return z


def read_img(filename, inputchannel):
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=inputchannel)


def get_dataset(train_glob, patchsize=256, batchsize=16, preprocess_threads=8):
    files = glob.glob(train_glob)
    if not files:
        raise RuntimeError(f"No training images found with glob "
                           f"'{train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: _parse_function_glob(x, patchsize), num_parallel_calls=preprocess_threads)
    dataset = dataset.batch(batchsize, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


def _parse_function_glob(x, patch_size=120):
    noisy = read_img(x, 3)
    noisy = tf.image.random_crop(noisy, [patch_size, patch_size, 3])
    noisy = tf.divide(tf.cast(noisy, tf.float32), 255.)
    idx = tf.random.uniform([], 0, 8, tf.int32)
    noisy = augment(noisy, idx)
    noisy = tf_normalize(noisy, patch_size)
    return noisy

def S2B(img, ratio):
    def fn0():
        return img

    def fn1():
        return tf.space_to_batch_nd(img, [2, 2], [[0, 0], [0, 0]])

    def fn2():
        return tf.space_to_batch_nd(img, [3, 3], [[0, 0], [0, 0]])

    def fn3():
        return tf.space_to_batch_nd(img, [4, 4], [[0, 0], [0, 0]])

    def fn4():
        return tf.space_to_batch_nd(img, [5, 5], [[0, 0], [0, 0]])

    return tf.switch_case(ratio - 1, {0: fn0, 1: fn1, 2: fn2, 3: fn3, 4: fn4})


def B2S(img, ratio):
    def fn0():
        return img

    def fn1():
        return tf.batch_to_space_nd(img, [2, 2], [[0, 0], [0, 0]])

    def fn2():
        return tf.batch_to_space_nd(img, [3, 3], [[0, 0], [0, 0]])

    def fn3():
        return tf.batch_to_space_nd(img, [4, 4], [[0, 0], [0, 0]])

    def fn4():
        return tf.batch_to_space_nd(img, [5, 5], [[0, 0], [0, 0]])

    return tf.switch_case(ratio - 1, {0: fn0, 1: fn1, 2: fn2, 3: fn3, 4: fn4})