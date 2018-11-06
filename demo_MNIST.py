import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import evaluation
import calibration

# import data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

# set parameters
learning_rate = 0.5
number_of_iterations = 500

# create the model
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

layer1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
logits = tf.layers.dense(layer1, 10)
predictions = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # training
    for _ in range(number_of_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={
            x: batch_xs,
            y: batch_ys
        })

    # evaluation
    accuracy = sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y: mnist.test.labels
    })

    valid_logits = sess.run(logits, feed_dict={
        x: mnist.test.images
    })

    valid_predictions = sess.run(predictions, feed_dict={
        x: mnist.test.images
    })

    sparse_labels = np.argmax(mnist.test.labels, axis=1)

    bins, average_confs, num_total = evaluation.get_bins(valid_predictions, sparse_labels)
    evaluation.plot_bins(bins, average_confs)

    ece = evaluation.get_ece(bins, average_confs, num_total)

    print("ECE before calibration: {:.2f}%".format(ece * 100))

    temp = calibration.temperature_scale(valid_logits, sess, sparse_labels, learning_rate=0.01, num_steps=50)
    scaled_logits = logits / temp

    scaled_predictions = tf.nn.softmax(scaled_logits)

    valid_predictions = sess.run(scaled_predictions, feed_dict={
        x: mnist.test.images
    })

    bins, average_confs, num_total = evaluation.get_bins(valid_predictions, sparse_labels)
    evaluation.plot_bins(bins, average_confs)

    ece = evaluation.get_ece(bins, average_confs, num_total)

    print()
    print("best temperature: {:.8f}".format(sess.run(temp)))
    print("ECE after calibration: {:.2f}%".format(ece * 100))

print("\nTrained for %d iterations" % number_of_iterations)
print("Accuracy: %.2f%%" % (accuracy * 100))