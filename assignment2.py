from __future__ import print_function
import numpy as np
import tensorflow as tf
import cPickle as pickle


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


num_steps = 801


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases.
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


batch_size = 128

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


'''
Additional Method here.
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


'''
One solution from online.
'''
batch_size = 128
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  print(lay1_train.get_shape())
  logits = tf.matmul(lay1_train, weights2) + biases2
  print(logits.get_shape())
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  print(train_prediction.get_shape())
  lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
  lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


'''
My solution here.
'''


batch_size = 40

# This method need to be improved.
def nn_logits(tf_dataset, keep_prob):
    weight_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    bias_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    hidden_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            tf.reshape(tf_dataset, [-1, image_size, image_size, 1]), weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
    # hidden_pool1 = tf.nn.max_pool(hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(hidden_conv1.get_shape())
    hidden_pool1 = tf.nn.max_pool(hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # print(hidden_pool1)
    # For Densely Connected Layer
    weight_fc1 = tf.Variable(tf.truncated_normal([14 * 14 * 32, 1024],stddev=0.1))
    bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    hidden_pool1_flat = tf.reshape(hidden_pool1, [-1, 14*14*32])
    # print(hidden_pool1_flat.get_shape())
    h_fc1 = tf.nn.relu(tf.matmul(hidden_pool1_flat, weight_fc1) + bias_fc1)
    # print(h_fc1.get_shape())

    # For the dropout

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Variables for softmax
    weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
    biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

    # Training computation.
    logits = tf.matmul(h_fc1_drop, weights) + biases
    # print(logits.get_shape())

    return logits

p_graph = tf.Graph()
with p_graph.as_default():
    # Input data. For training data, we use a placeholder that will be fed
    # at run time with a training minibatch
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    keep_prob = tf.placeholder(tf.float32)

    weight_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
    bias_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    train_hidden_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            tf.reshape(tf_train_dataset, [-1, image_size, image_size, 1]), weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
    # print(hidden_conv1.get_shape())
    train_hidden_pool1 = tf.nn.max_pool(train_hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # print(hidden_pool1)
    # For Densely Connected Layer
    weight_fc1 = tf.Variable(tf.truncated_normal([14 * 14 * 32, 1024],stddev=0.1))
    bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    train_hidden_pool1_flat = tf.reshape(train_hidden_pool1, [-1, 14*14*32])
    # print(hidden_pool1_flat.get_shape())
    train_h_fc1 = tf.nn.relu(tf.matmul(train_hidden_pool1_flat, weight_fc1) + bias_fc1)
    # print(h_fc1.get_shape())

    # For the dropout
    train_h_fc1_drop = tf.nn.dropout(train_h_fc1, keep_prob)
    # Variables for softmax
    weights = tf.Variable(tf.truncated_normal([1024, num_labels]))
    biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

    # Training computation.
    train_logits = tf.matmul(train_h_fc1_drop, weights) + biases
    # print(logits.get_shape())

    # train_logits = nn_logits(tf_train_dataset, keep_prob)

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
    loss = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(tf.clip_by_value(tf.nn.softmax(train_logits), 1e-10, 1.0)), reduction_indices=[1]))

    # Optimizer
    # Alpha = 0.5 or 1e-3
    optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)

    # Predictions for the training, validation, and test data
    train_prediction = tf.nn.softmax(train_logits)

    valid_hidden_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            tf.reshape(tf_valid_dataset, [-1, image_size, image_size, 1]), weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
    valid_hidden_pool1 = tf.nn.max_pool(valid_hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    valid_hidden_pool1_flat = tf.reshape(valid_hidden_pool1, [-1, 14*14*32])
    valid_h_fc1 = tf.nn.relu(tf.matmul(valid_hidden_pool1_flat, weight_fc1) + bias_fc1)
    # For the dropout
    valid_h_fc1_drop = tf.nn.dropout(valid_h_fc1, 1.0)
    valid_logits = tf.matmul(valid_h_fc1_drop, weights) + biases

    test_hidden_conv1 = tf.nn.relu(
        tf.nn.conv2d(
            tf.reshape(tf_test_dataset, [-1, image_size, image_size, 1]), weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
    test_hidden_pool1 = tf.nn.max_pool(test_hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    test_hidden_pool1_flat = tf.reshape(test_hidden_pool1, [-1, 14*14*32])
    test_h_fc1 = tf.nn.relu(tf.matmul(test_hidden_pool1_flat, weight_fc1) + bias_fc1)
    test_h_fc1_drop = tf.nn.dropout(test_h_fc1, 1.0)
    test_logits = tf.matmul(test_h_fc1_drop, weights) + biases

    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)


num_steps = 9001

with tf.Session(graph=p_graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    train_feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=train_feed_dict)

    if (step % 100 == 0):
        print(l)
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))

  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

