import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import cv2 as cv
from skimage import feature
from sklearn.utils import shuffle


image_size = 28
number_class = 10
image_shape = (image_size, image_size)
image_size_flat = image_size*image_size


def read_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    label = train['label']
    train = train.drop("label", axis=1)
    # print(train.shape, label.shape)
    return label, train, test


def one_hot(y):
    m = []
    for i in y:
        xx = np.zeros((10))
        xx[i] = 1
        m.append(xx)
    return np.array(m)


def weights(shape, mu=0, std=0.05):
    return tf.Variable(tf.truncated_normal(shape, mu, std), dtype=tf.float32)


def biases(length, value=0):
    return tf.Variable(tf.constant(value, dtype=tf.float32, shape=[length]), dtype=tf.float32)


def new_fc_layer(input, num_inputs, num_outputs, activation="no"):
    Wi = weights(shape=[num_inputs, num_outputs])
    Bi = biases(length=num_outputs)
    layer = tf.matmul(input, Wi) + Bi
    if activation=="softmax":
        layer = tf.nn.softmax(input)
    elif activation=="relu":
        layer = tf.nn.relu(layer)
    elif activation=="sigmoid":
        layer = tf.nn.sigmoid(layer)
    elif activation=="tanh":
        layer = tf.nn.tanh(layer)
    elif activation=="leaky":
        layer = tf.nn.leaky_relu(layer)
    return layer, Wi, Bi


def model(X_train, Y_train, X_test, learning_rate=0.875):
    x = tf.placeholder(tf.float32, [None, image_size_flat], name="x")
    y = tf.placeholder(tf.float32, [None, number_class], name='y')
    layer1, W1, B1 = new_fc_layer(input=x, num_inputs=X_train.shape[1], num_outputs=number_class, activation="no")
    layer2, _, _ = new_fc_layer(input=layer1, num_inputs=number_class, num_outputs=number_class, activation="softmax")
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer1, labels=y)
    regularizer = tf.nn.l2_loss(W1)
    # beta = 1e-7
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(layer2, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def print_accuracy(feed, val=True):
        acc = session.run(accuracy, feed_dict=feed)
        if val:
            print("Accuracy on val-set: {0:.3%}".format(acc))
        else:
            print("Accuracy on train-set: {0:.3%}".format(acc))
        return acc

    def optimize(num_iterations, learning_rate):
        acc = []
        abb = []
        for i in range(num_iterations):
            print("===============================================")
            print("epoch ", i+1)
            print("learning_rate", learning_rate)
            print("===============================================")
            if i % 50 is 0 and i is not 0:
                rng_state = np.random.get_state()
                np.random.shuffle(X_train)
                np.random.set_state(rng_state)
                np.random.shuffle(Y_train)

            # X_train, Y_train = shuffle(X_train, Y_train)
            feed_dict_train = {x: X_train[:41000, :], y: Y_train[:41000]}
            feed_dict_val = {x: X_train[41000:, :], y: Y_train[41000:]}
            feed_dict = {x: X_train, y: Y_train}
            session.run(optimizer, feed_dict=feed_dict)
            acc.append(print_accuracy(feed_dict))
            # abb.append(print_accuracy(feed_dict_train, False))
            if i % 25 is 0 and i is not 0:
                learning_rate /= 2
                if i is 50:
                    learning_rate=0.875
        # plt.plot(np.arange(0, len(abb)), abb, 'r')
        plt.plot(np.arange(0, len(acc)), acc, 'b')
        plt.show()
    optimize(100, learning_rate)
    preds = session.run(tf.argmax(layer2, axis=1), feed_dict={x: X_test})
    pd.DataFrame({"ImageId": list(range(1, len(preds) + 1)), "Label": preds}).to_csv("MLP_baby.csv", index=False, header=True)
    session.close()


def main():
    class_label, train_image, test_image = read_data()
    X_train = train_image.values
    Y_train = class_label.values
    # X_train_normalized = (X_train-np.mean(X_train))/np.std(X_train)
    # print(np.histogram(Y_train[41000:]))
    X_test = test_image.values
    Y_train = one_hot(Y_train)
    model(X_train=X_train, Y_train=Y_train, X_test=X_test)



if __name__ == '__main__':
    main()