import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Paramter tuning
input_nodes = 784
output_nodes = 10
hidden1_nodes = 100
hidden2_nodes = 80
batch_size = 50
learning_rate_init = 0.9
lr_decay = 0.9
beta = 0.0005
epochs = 5000
av_decay = 0.9
dropout_prob = 0.8


def output(input_array, av_class, w1, b1, w2, b2, w3, b3):
    # Using ReLU as the activation function
    if av_class == None:
        hidden1 = tf.nn.relu(tf.add(tf.matmul(input_array, w1), b1))
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, w2), b2))
        return tf.add(tf.matmul(hidden2, w3), b3)
    else:
        hidden1 = tf.nn.relu(tf.add(tf.matmul(input_array, av_class.average(w1)), av_class.average(b1)))
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, av_class.average(w2)), av_class.average(b2)))
        return (tf.addtf.matmul(hidden2, av_class.average(w3)), av_class.average(b3))

def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_nodes], name='x-input') # input placeholder - 28 x 28 pixels = 784
    y_ = tf.placeholder(tf.float32, [None, output_nodes], name='y-input') # output placeholder - 10 digits

    # Declaring and randomly initializing the number of weights and biases for each layer.
    
    # Hidden layer 1
    w1 = tf.Variable(tf.truncated_normal([input_nodes, hidden1_nodes], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_nodes]))

    # Hidden layer 2
    w2 = tf.Variable(tf.truncated_normal([hidden1_nodes, hidden2_nodes], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_nodes]))

    # Output layer
    w3 = tf.Variable(tf.truncated_normal([hidden2_nodes, output_nodes], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[output_nodes]))

    y = output(x, None, w1, b1, w2, b2, w3, b3)

    y = tf.nn.dropout(y, dropout_prob) # Applying dropout on the output

    global_step = tf.Variable(0, trainable=False)
    var_av = tf.train.ExponentialMovingAverage(av_decay, global_step)
    var_av_op = var_av.apply(tf.trainable_variables())
    average_y = output(x, None, w1, b1, w2, b2, w3, b3)
    
    # output layer - Using Softmax for probabilities
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(beta)
    reg = regularizer(w1) + regularizer(w2)
    loss = cross_entropy_mean + reg

    learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, mnist.train.num_examples/batch_size, lr_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, var_av_op]):
        train_op = tf.no_op(name = 'train')


    prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # training and checking accuracies
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_data = {x: mnist.validation.images, y_:mnist.validation.labels}
        test_data = {x:mnist.test.images, y_:mnist.test.labels}
        for i in range(epochs):
            if (i%500==0): # Checking validation accuracy every 500 epochs
                validate_acc = sess.run(accuracy, feed_dict=validate_data)
                print("Epochs: %d. Validation accuracy = %g"%(i, validate_acc))

            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x:xs, y_:ys})

        test_acc = sess.run(accuracy, feed_dict=test_data)
        print("Epochs: %d. Testing accuracy = %g"%(epochs, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

tf.app.run()