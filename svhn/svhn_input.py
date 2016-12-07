from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import scipy.io as sio
import numpy as np

# Process images of this size. A number which make impact to entire model
# architecture.
IMAGE_SIZE = 24

# Global constants describing the CIFAT-10 data set
NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_svhn_mat_files(filenames):
    data = None
    labels = None

    # Dict Keys from mat files
    # mat file keys :['y', 'X', '__version__', '__header__', '__globals__' ]

    for mat_file in filenames:
        if not tf.gfile.Exists(mat_file):
            raise ValueError('Failed to find file: ' + mat_file)
        print(mat_file)
        with open(mat_file, 'rb') as m:
            save = sio.loadmat(m)
            s_data = save['X']
            s_labels = save['y']
            del save
            print('data set', s_data.shape, s_labels.shape)

            data = np.append(data, s_data, axis=0) if data is not None else s_data
            labels = np.append(labels, s_labels, axis=0) if labels is not None else s_labels

    print('Data set: ', data.shape, labels.shape)
    data_t = data.transpose(3, 0, 1, 2)
    return data_t, labels


def read_svhn_to_queue(filenames):

    data, labels= read_svhn_mat_files(filenames)

    data_queue = tf.train.input_producer(data, shuffle=False)
    labels_queue = tf.train.input_producer(labels, shuffle=False)

    return data_queue, labels_queue


def read_svhn_reader(data_q, labels_q):
    return data_q.dequeue(), labels_q.dequeue()


def read_svhn(filenames):
    class SVHNRecord(object):
        pass

    result = SVHNRecord()

    # Dimensions of the images in the svhn dataset.
    # See http://ufldl.stanford.edu/housenumbers/ for a description of the
    # input format.
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    data_q, label_q = read_svhn_to_queue(filenames)
    data, label = read_svhn_reader(data_q, label_q)
    print(data.get_shape(), data.dtype)
    print(label.get_shape(), label.dtype)

    result.label = tf.cast(label, tf.int32)
    # depth_major = tf.reshape(data, [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    # result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    result.uint8image = data
    print(data.get_shape(), data.dtype)
    print(result.label.get_shape(), result.label.dtype)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
            #capacity=1280 + 3 * batch_size,
            #min_after_dequeue=1280
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    print ('mark: _generate_image_and_label_batch')
    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'train_32x32.mat')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    print(filenames)

    # Read examples from files in the filename queue.
    read_input = read_svhn(filenames)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    print ('mark: distorted_inputs')
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):

    if not eval_data:
        filenames = [os.path.join(data_dir, 'train_32x32.mat')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_32x32.mat')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Read examples from files in the filename queue.
    read_input = read_svhn(filenames)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
