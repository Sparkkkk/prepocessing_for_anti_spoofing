import tensorlayer as tl
import tensorflow as tf
import cv2
import numpy as np
import sys
import os
from extract_frame.extract_face_from_video import crop_face
from extract_frame.utility import create_pb_from_ckpt, convert_pb_to_coreml


path = sys.path[0]
model_path = os.path.join(path, '..', 'model')
data_path = os.path.join(path, '..', 'data')
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(data_path, 'image')

# prepare data
# img0 = cv2.imread(os.path.join(image_path, 'my_image_real.jpg'))
# img1 = cv2.imread(os.path.join(image_path, 'my_image_fake.jpg'))
# img2 = cv2.imread(os.path.join(image_path, 'image_real_1.jpg'))
# img3 = cv2.imread(os.path.join(image_path, 'image_fake_2.jpg'))
# X_test = np.ndarray([4, 112, 112, 3])
# X_test[0] = crop_face(img0)
# X_test[1] = crop_face(img1)
# X_test[2] = crop_face(img2)
# X_test[3] = crop_face(img3)
# y_test = np.array([0, 1, 0, 1])


# define network
def network_97(x):
    t_network = tl.layers.InputLayer(x, name='input_layer')
    t_network = tl.layers.Conv2d(t_network, 5, (5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    t_network = tl.layers.MaxPool2d(t_network, (3, 3), (3, 3), 'SAME', name='max_pooling2d')
    t_network = tl.layers.Conv2d(t_network, 20, (5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    t_network = tl.layers.MaxPool2d(t_network, (3, 3), (3, 3), name='maxpool2d')
    t_network = tl.layers.FlattenLayer(t_network, name='flatten')
    t_network = tl.layers.DropoutLayer(t_network, keep=0.5, name='dropout1')
    t_network = tl.layers.DenseLayer(t_network, 256, act=tf.nn.relu, name='dense1')
    t_network = tl.layers.DropoutLayer(t_network, keep=0.5, name='dropout2')
    t_network = tl.layers.DenseLayer(t_network, 2, act=tf.identity, name='output_layer')
    return t_network


def network_99(x):
    # define network
    t_network = tl.layers.InputLayer(x, name='input_layer')
    t_network = tl.layers.Conv2d(t_network, 20, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1')
    t_network = tl.layers.MeanPool2d(t_network, (3, 3), (3, 3), name='max_pooling2d')
    t_network = tl.layers.Conv2d(t_network, 30, (3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2')
    t_network = tl.layers.MeanPool2d(t_network, (3, 3), (3, 3), name='maxpool2d')
    t_network = tl.layers.FlattenLayer(t_network, name='flatten')
    t_network = tl.layers.DropoutLayer(t_network, keep=0.5, name='dropout1')
    t_network = tl.layers.DenseLayer(t_network, 256, act=tf.nn.relu, name='dense1')
    t_network = tl.layers.DropoutLayer(t_network, keep=0.5, name='dropout2')
    t_network = tl.layers.DenseLayer(t_network, 2, act=tf.identity, name='output_layer')
    return t_network


def evaluate_model(model_name):
    # X_test = tl.files.load_npy_to_any(dataset_path, 'real_face1.npy')
    # y_test = np.zeros(X_test.shape[0])
    X_test = tl.files.load_npy_to_any(dataset_path, 'fake_face.npy')
    y_test = np.ones(X_test.shape[0])
    # X_test = np.append(X_real, X_fake, axis=0)
    # assert X_test.shape == (X_test.shape[0], 112, 112, 3)
    # y_test = np.append(y_real, y_fake, axis=0)

    # create a session for tf
    sess = tf.InteractiveSession()

    # define placeholder
    batch_size = 100
    x = tf.placeholder(tf.float32, shape=[None, 112, 112, 3], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    network = network_99(x)

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
    tl.utils.test(sess, network, accuracy, X_test, y_test, x, y_, batch_size=None, cost=cost)

    # create a pb file to store graph and parameter
    # create_pb_from_ckpt('LeNet_anti_spoofing_97', 'Softmax')


if __name__ == '__main__':
    evaluate_model(model_name='LeNet_anti_spoofing_99.npz')




