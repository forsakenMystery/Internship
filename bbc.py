import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img_size = 28
num_classes = 10
img_shape = (img_size, img_size)
img_size_flat = img_size*img_size
num_channels = 1
batch_size = 64
l2_regulizer = 0.001

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36

fc_size = 128


def new_weights(shape, name):
    return tf.get_variable(name=name, initializer=tf.contrib.layers.xavier_initializer(), shape=shape)


def new_biases(length, name):
    return tf.get_variable(name, shape=[length])


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    s = np.random.randint(0, 100)
    weights = new_weights(shape, 'conv'+str(s))
    biases = new_biases(num_filters, 'bias'+str(s))
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    s = np.random.randint(0, 100)
    weights = new_weights(shape=[num_inputs, num_outputs], name='w'+str(s))
    biases = new_biases(length=num_outputs, name='b'+str(s))
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer, weights


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

print(layer_conv1)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)

print(layer_conv2)

layer_flat, num_features = flatten_layer(layer_conv2)

print(layer_flat)
print(num_features)

layer_fc1, weights3 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

print(layer_fc1)

layer_fc2, weights4 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

print(layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.add(tf.reduce_mean(cross_entropy), tf.multiply(l2_regulizer/batch_size, tf.add(tf.add(tf.add(tf.nn.l2_loss(weights_conv1), tf.nn.l2_loss(weights_conv2)), tf.nn.l2_loss(weights3)), tf.nn.l2_loss(weights4))))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())


def optimize(num_iterations):
    accu = []
    vali = []
    for i in range(num_iterations):
        for j in range(total_batches):
            index_front = j * batch_size
            index_end = (j + 1) * batch_size if (j + 1) * batch_size < x_train.shape[0] else x_train.shape[0]
            X_batch = x_train[index_front:index_end, :]
            Y_batch = y_train[index_front:index_end, :]
            # print("bitch")
            feed_dict_train = {x: X_batch, y_true: Y_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        if True:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
            feed_dict_val = {x: x_val, y_true: y_val}
            val = session.run(accuracy, feed_dict=feed_dict_val)
            msg = "Optimization Iteration: {0:>6}, Validation Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, val))
            accu.append(acc)
            vali.append(val)
    plt.plot(np.arange(0, len(accu)), accu, 'r')
    plt.plot(np.arange(0, len(vali)), vali, 'b')
    plt.show()


def convert_to_one_hot(labels, depth):
    one_hot_matrix = tf.one_hot(labels, depth, axis=0)
    sess = tf.Session()
    one_hot_labels = sess.run(one_hot_matrix)
    sess.close()
    return one_hot_labels


def load_dataset():
    x_train = pd.read_csv('train.csv').as_matrix()
    x_test = np.divide(pd.read_csv('test.csv').as_matrix(), 255)
    y_train = x_train[:, 0].reshape(1, x_train.shape[0])
    print(y_train.shape)
    y_train = np.squeeze(convert_to_one_hot(y_train, int(np.amax(y_train) + 1))).T
    print(y_train.shape)
    x_train = np.divide(x_train[:, 1:], 255)
    validation_images = x_train[0:2048, :]
    validation_labels = y_train[0:2048, :]
    x_train = x_train[2048:, :]
    y_train = y_train[2048:, :]
    return x_train, y_train, validation_images, validation_labels, x_test


x_train, y_train, x_val, y_val, x_test = load_dataset()
print(x_train.shape)
print(y_train.shape)
total_batches = np.ceil(x_train.shape[0] / batch_size).astype(np.int32)
optimize(50)
print(x_test.shape)
print(x_train.shape)
predict = session.run(y_pred_cls, feed_dict={x: x_test})
print(predict.shape)
print(predict)
image_id = np.arange(1, len(predict) + 1, 1)
image_id = image_id.reshape(len(predict), 1)
np.savetxt('kosagam.csv', np.c_[image_id, predict], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
