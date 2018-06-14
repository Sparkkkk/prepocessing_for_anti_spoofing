import tensorlayer as tl
import tensorflow as tf
import cv2
import numpy as np
import sys
import os
from src.extract_face_from_video import crop_face
from src.utility import create_pb_from_ckpt, convert_pb_to_coreml


path = sys.path[0]
model_path = os.path.join(path, '..', 'model')
data_path = os.path.join(path, '..', 'data')
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(data_path, 'image')


# define network
def network_97(x):
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, 5, (5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    network = tl.layers.MaxPool2d(network, (3, 3), (3, 3), 'SAME', name='max_pooling2d')
    network = tl.layers.Conv2d(network, 20, (5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    network = tl.layers.MaxPool2d(network, (3, 3), (3, 3), name='maxpool2d')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='dense1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.DenseLayer(network, 2, act=tf.identity, name='output_layer')
    return network


def network_99(x):
    # define network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, 20, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='max_pooling2d')
    network = tl.layers.Conv2d(network, 30, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='maxpool2d')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='dense1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.DenseLayer(network, 2, act=tf.identity, name='output_layer')
    return network


def network_992(x):
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, 20, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='max_pooling2d')
    network = tl.layers.Conv2d(network, 30, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='maxpool2d')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='dense1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 128, act=tf.nn.relu, name='dense2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.DenseLayer(network, 2, act=tf.identity, name='output_layer')
    return network


def network_vgg11_813(x):
    # define network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, 64, (7, 7), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), name='max_pooling2d')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.Conv2d(network, 128, (5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), name='maxpool2d')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.Conv2d(network, 256, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3')
    network = tl.layers.Conv2d(network, 256, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4')
    network = tl.layers.MaxPool2d(network, (2, 2), (2, 2), name='maxpool2d')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout3')
    network = tl.layers.Conv2d(network, 512, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5')
    network = tl.layers.Conv2d(network, 512, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv6')

    return network


def network_99_three_dense(x):
    # define network
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2d(network, 20, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='max_pooling2d')
    network = tl.layers.Conv2d(network, 30, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    network = tl.layers.MeanPool2d(network, (3, 3), (3, 3), name='maxpool2d')
    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 512, act=tf.nn.relu, name='dense1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout1')
    network = tl.layers.DenseLayer(network, 256, act=tf.nn.relu, name='dense2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.DenseLayer(network, 128, act=tf.nn.relu, name='dense3')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='dropout2')
    network = tl.layers.DenseLayer(network, 2, act=tf.identity, name='output_layer')
    return network


def get_X_y_test(npy_file_name, second_file_name=None):
    dictionary = tl.files.load_npy_to_any(dataset_path, '%s.npy' % npy_file_name)
    X_test = dictionary['X']
    y_test = dictionary['y']
    if second_file_name is not None:
        dictionary_2 = tl.files.load_npy_to_any(dataset_path, '%s.npy' % second_file_name)
        X_test = np.append(X_test, dictionary_2['X'], axis=0)
        y_test = np.append(y_test, dictionary_2['y'], axis=0)
    return (X_test, y_test)


def evaluate_model(X_test, y_test, model_name, test_accurary=True):

    # create a session for tf
    sess = tf.InteractiveSession()

    # define placeholder
    batch_size = 100
    x = tf.placeholder(tf.float32, shape=[None, 112, 112, 3], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    network = network_99_three_dense(x)

    # print network information
    # network.print_params()
    # network.print_layers()

    # define cost function and metric
    y = network.outputs
    cost = tl.cost.cross_entropy(y, y_, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    y_op = tf.argmax(tf.nn.softmax(y), 1)

    tl.files.load_and_assign_npz(sess, name=os.path.join(model_path, model_name), network=network)
    # tl.files.save_ckpt(sess=sess, mode_name='LeNet_anti_spoofing_97.ckpt', save_dir='model', printable=True)

    # tf.train.write_graph(sess.graph_def, 'model', 'LeNet_anti_spoofing_97.pbtxt')
    if test_accurary:
        tl.utils.test(sess, network, accuracy, X_test, y_test, x, y_, batch_size=None, cost=cost)
        sess.close()
    else:
        return sess, network, x, y_op

    # create a pb file to store graph and parameter
    # create_pb_from_ckpt('LeNet_anti_spoofing_97', 'Softmax')



if __name__ == '__main__':
    evaluate_model('test', model_name='LeNet_anti_spoofing_99.npz')




