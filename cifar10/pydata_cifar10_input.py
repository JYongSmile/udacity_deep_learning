from __future__ import print_function
import os
import tensorflow as tf
import cPickle as pickle
import numpy as np

from PIL import Image
from scipy import ndimage

# Global constants describing the CIFAR-10 data set
# CIFAR10 image size of 32x32. will distort to 24x24
IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# train_data_queue = None
# train_labels_queue = None
# train_f_names_queue = None


def read_cifar10_python_pickles(filenames):
    data = None
    labels = None
    f_names = None
    # Dict Keys from pickle files
    # ['data', 'labels', 'batch_label', 'filenames']
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
                 for i in xrange(1, 6)]"""
    for pickle_file in filenames:
        if not tf.gfile.Exists(pickle_file):
            raise ValueError('Failed to find file: ' + pickle_file)
        with open(pickle_file, 'rb') as p:
            save = pickle.load(p)
            s_data = save['data']
            s_labels = np.array(save['labels'])
            s_f_names = np.array(save['filenames'])
            del save
            print('data set', s_data.shape, s_labels.shape)

            data = np.append(data, s_data, axis=0) if data is not None else s_data
            labels = np.append(labels, s_labels, axis=0) if labels is not None else s_labels
            f_names = np.append(f_names, s_f_names, axis=0) if f_names is not None else s_f_names
    print('Data set: ', data.shape, len(labels))
    return data, labels, f_names


def read_cifar10_python_pickle(filename):
    if not tf.gfile.Exists(filename):
        raise ValueError('Failed to find file: ' + filename)
    with open(filename, 'rb') as p:
        save = pickle.load(p)
        data = save['data']
        labels = np.array(save['labels'])
        f_names = np.array(save['filenames'])
        del save
        print('data set', data.shape, labels.shape)

    return data, labels, f_names


def read_cifar10_to_queue(filenames):

    data, labels, f_names = read_cifar10_python_pickles(filenames)

    data_queue = tf.train.input_producer(data, shuffle=False)
    labels_queue = tf.train.input_producer(labels, shuffle=False)
    f_names_queue = tf.train.input_producer(f_names, shuffle=False)

    return data_queue, labels_queue, f_names_queue


def read_cifar10_reader(data_q, labels_q):
    return data_q.dequeue(), labels_q.dequeue()


def read_cifar10(filenames):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3

    data_q, label_q, _ = read_cifar10_to_queue(filenames)
    data, label = read_cifar10_reader(data_q, label_q)
    print(data.get_shape(), data.dtype)
    print(label.get_shape(), label.dtype)
    result.label = tf.cast(label, tf.int32)
    depth_major = tf.reshape(data, [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    print(depth_major.get_shape(), depth_major.dtype)
    print(result.label.get_shape(), result.label.dtype)
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    #tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.
        Args:
            data_dir: Path to the CIFAR-10 data directory.
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.

    """

    filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1,6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    read_input = read_cifar10(filenames)
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
    # min_queue_examples: train(50000*0.4=20000) eval(10000*0.4=4000)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    read_input = read_cifar10(filenames)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_whitening(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def test_inputs(data_dir):
    filenames = [os.path.join(data_dir, 'test_batch')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    read_input = read_cifar10(filenames)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_whitening(resized_image)

    return_image = tf.reshape(float_image, [-1, height, width, 3])
    return return_image, read_input.label


def test_input_process(image, label=None):
    reshaped_image = tf.cast(image, tf.float32)
    if label is not None:
        label = tf.cast(label, tf.int32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_whitening(resized_image)

    return_image = tf.reshape(float_image, [-1, height, width, 3])
    return return_image, label


def read_n_preprocess_external_input(image_file):
    temp_image = Image.open(image_file)
    height = temp_image.size[1]
    width = temp_image.size[0]
    if height > width:
        temp_image = temp_image.crop((0, (height - width)/2, width, (height + width)/2))
    elif height < width:
        temp_image = temp_image.crop(((width - height)/2, 0, (height + width)/2, height))
    temp_image.thumbnail((32, 32), Image.ANTIALIAS)
    temp_image_arr = np.array(temp_image)
    return temp_image_arr


