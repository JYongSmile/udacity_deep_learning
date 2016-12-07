# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
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
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


'''
Begin with logistic model.
'''
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
    # One more placeholder for param of l2 regularization
    lambda_regul = tf.placeholder(tf.float32)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    # Loss with l2 regularization
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + lambda_regul * tf.nn.l2_loss(weights)
    )

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


'''
One hidden layer with 1024 nodes.
'''

num_hidden_nodes = 1024

oneHiddenLayerGraph = tf.Graph()
with oneHiddenLayerGraph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  # meta param for l2 regularization.
  lambda_regul = tf.placeholder(tf.float32)
  global_step = tf.Variable(0)

  # Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))

  # Training computation.
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(lay1_train, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
      lambda_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))

  # Optimizer.
  # Optimizer.
  learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
  valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
  lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
  test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)


'''
Wrap into a funtion.
'''


def tf_deep_nn(regular=False, drop_out=False, lrd=False, layer_cnt=2):
    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        hidden_node_count = 1024
        # start weight
        hidden_stddev = np.sqrt(2.0 / 784)
        weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_node_count], stddev=hidden_stddev))
        biases1 = tf.Variable(tf.zeros([hidden_node_count]))
        # middle weight
        weights = []
        biases = []
        hidden_cur_cnt = hidden_node_count
        for i in range(layer_cnt - 2):
            if hidden_cur_cnt > 2:
                hidden_next_cnt = int(hidden_cur_cnt / 2)
            else:
                hidden_next_cnt = 2
            hidden_stddev = np.sqrt(2.0 / hidden_cur_cnt)
            weights.append(tf.Variable(tf.truncated_normal([hidden_cur_cnt, hidden_next_cnt], stddev=hidden_stddev)))
            biases.append(tf.Variable(tf.zeros([hidden_next_cnt])))
            hidden_cur_cnt = hidden_next_cnt
        # first wx + b
        y0 = tf.matmul(tf_train_dataset, weights1) + biases1
        # first relu
        hidden = tf.nn.relu(y0)
        hidden_drop = hidden
        # first DropOut
        keep_prob = 0.5
        if drop_out:
            hidden_drop = tf.nn.dropout(hidden, keep_prob)
        # first wx+b for valid
        valid_y0 = tf.matmul(tf_valid_dataset, weights1) + biases1
        valid_hidden = tf.nn.relu(valid_y0)
        # first wx+b for test
        test_y0 = tf.matmul(tf_test_dataset, weights1) + biases1
        test_hidden = tf.nn.relu(test_y0)

        # middle layer
        for i in range(layer_cnt - 2):
            y1 = tf.matmul(hidden_drop, weights[i]) + biases[i]
            hidden_drop = tf.nn.relu(y1)
            if drop_out:
                keep_prob += 0.5 * i / (layer_cnt + 1)
                hidden_drop = tf.nn.dropout(hidden_drop, keep_prob)

            y0 = tf.matmul(hidden, weights[i]) + biases[i]
            hidden = tf.nn.relu(y0)

            valid_y0 = tf.matmul(valid_hidden, weights[i]) + biases[i]
            valid_hidden = tf.nn.relu(valid_y0)

            test_y0 = tf.matmul(test_hidden, weights[i]) + biases[i]
            test_hidden = tf.nn.relu(test_y0)

        # last weight
        weights2 = tf.Variable(tf.truncated_normal([hidden_cur_cnt, num_labels], stddev=hidden_stddev / 2))
        biases2 = tf.Variable(tf.zeros([num_labels]))
        # last wx + b
        logits = tf.matmul(hidden_drop, weights2) + biases2

        # predicts
        logits_predict = tf.matmul(hidden, weights2) + biases2
        valid_predict = tf.matmul(valid_hidden, weights2) + biases2
        test_predict = tf.matmul(test_hidden, weights2) + biases2

        l2_loss = 0
        # enable regularization
        if regular:
            l2_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)
            for i in range(len(weights)):
                l2_loss += tf.nn.l2_loss(weights[i])
                # l2_loss += tf.nn.l2_loss(biases[i])
            # beta = 0.25 / batch_size
            beta = 1e-5
            l2_loss *= beta
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + l2_loss

        # Optimizer.
        if lrd:
            cur_step = tf.Variable(0, trainable=False)  # count the number of steps taken.
            starter_learning_rate = 0.4
            learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 100000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits_predict)
        valid_prediction = tf.nn.softmax(valid_predict)
        test_prediction = tf.nn.softmax(test_predict)

    num_steps = 20001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            offset_range = train_labels.shape[0] - batch_size
            offset = (step * batch_size) % offset_range
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if step % 500 == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


'''
Graph with CNN layer and full connected layer

TODO: To wrap it into a method.
'''


def tf_simple_cnn(batch_size=128, regular=False, drop_out=False, lrd=False, layer_cnt=2):
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
        lambda_regul = tf.placeholder(tf.float32)
        global_step = tf.Variable(0)

        hidden_stddev = np.sqrt(2.0 / 784)

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

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels)
                              + lambda_regul * (tf.nn.l2_loss(weight_conv1) + tf.nn.l2_loss(weight_fc1) + tf.nn.l2_loss(weights))
                            )
        '''
        loss = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(tf.clip_by_value(tf.nn.softmax(train_logits), 1e-10, 1.0)), reduction_indices=[1])
                        + lambda_regul * (tf.nn.l2_loss(weight_conv1) + tf.nn.l2_loss(weight_fc1) + tf.nn.l2_loss(weights))
                              )
        '''
        # Optimizer
        # Alpha = 0.5 or 1e-3
        learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.65, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

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


    # Running session

    num_steps = 3001

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
            # feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, lambda_regul: 1e-3}
            # If using one CNN layer
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, lambda_regul: 1e-3, keep_prob: 0.5}

            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))




'''
batch_size = 128
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
    lambda_regul = tf.placeholder(tf.float32)
    global_step = tf.Variable(0)

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
    loss = tf.reduce_mean(-tf.reduce_sum(tf_train_labels * tf.log(tf.clip_by_value(tf.nn.softmax(train_logits), 1e-10, 1.0)), reduction_indices=[1])
                    + lambda_regul * (tf.nn.l2_loss(weight_conv1) + tf.nn.l2_loss(weight_fc1) + tf.nn.l2_loss(weights))
                          )

    # Optimizer
    # Alpha = 0.5 or 1e-3
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

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


# Running session

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
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, lambda_regul: 1e-3}
        # If using one CNN layer
        # feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, lambda_regul: 1e-3, keep_prob: 0.5}

        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
'''
