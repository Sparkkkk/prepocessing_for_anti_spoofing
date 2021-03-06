import cv2
import tensorlayer as tl
from tensorflow.python.tools.freeze_graph import freeze_graph
from core import Detector
from core import Alignment
import tfcoreml
import numpy as np
import sys
import os

path = sys.path[0]
data_path = os.path.join(path, '..', 'data')
image_path = os.path.join(data_path, 'image')

# size = '112, 112'
detector = Detector()
aligner = Alignment()


def create_pb_from_ckpt(model_name, output_node):
    # convert to a pb file
    # Graph definition file, stored as protobuf TEXT
    graph_def_file = 'model_new/%s.pbtxt' % model_name
    # Trained model's checkpoint name
    checkpoint_file = 'model_new/%s.ckpt' % model_name
    # Frozen model's output name
    frozen_model_file = 'model_new/%s.pb' % model_name
    # Output nodes
    output_node_names = output_node

    # Call freeze graph
    freeze_graph(input_graph=graph_def_file,
                 input_saver="",
                 input_binary=False,
                 input_checkpoint=checkpoint_file,
                 output_node_names=output_node_names,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 output_graph=frozen_model_file,
                 clear_devices=True,
                 initializer_nodes="")


def convert_pb_to_coreml(path_pb, path_coreml, input_shape, tensor_name):
    coreml_model = tfcoreml.convert (
        tf_model_path=path_pb,
        mlmodel_path=path_coreml,
        image_input_names='data',
        input_name_shape_dict=input_shape,
        output_feature_names=tensor_name
    )
    return coreml_model


def evaluate_coreml(coreml_model):
    # Provide CoreML model with a dictionary as input. Change ':0' to '__0'
    # as Swift / Objective-C code generation do not allow colons in variable names
    coreml_inputs = {'Placeholder__0': np.random.rand(1, 1, 112, 112, 3)}  # (sequence_length=1,batch=1,channels=784)
    coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=False)
    return coreml_output


def crop_face_with_box(frame, size='112, 112'):
    box = detector.detect_face(frame)
    if box is None:
        return None, None
    # box[0, 0:4] = padding(3, box[0, 0:4], frame.shape)
    face_ndarray = aligner.warp_image(frame, box[0, 0:4], image_size=size)
    # face_ndarray = cv2.cvtColor(face_ndarray, cv2.COLOR_BGR2RGB)
    # face_ndarray = np.transpose(face_ndarray, (2, 0, 1))
    return face_ndarray, box


def padding(percentage, box, size):
    left = box[0]
    bottom = box[1]
    right = box[2]
    top = box[3]

    width = right - left
    height = bottom - top

    left = left - width * percentage / 2
    right = right + width * percentage / 2
    bottom = bottom + height * percentage / 2
    top = top - height * percentage / 2

    if left < 0:
        left = 0
    if right > size[0]:
        right = size[0]
    if top < 0:
        top = 0
    if bottom > size[1]:
        bottom = size[1]

    return np.asarray([left, bottom, right, top])


def min_max_regions(regions):
    left = np.min(regions[:, 0])
    bottom = np.max(regions[:, 1])
    right = np.max(regions[:, 2])
    top = np.min(regions[:, 3])

    return np.array([left, bottom, right, top])


# def crop_aligned_face(frame):
#     box = detector.detect_face(frame)
#     align = aligner.align_faces_without_transpose(frame, box)[0]
#
#     return align

def read_npy_file(name):
    return tl.files.load_npy_to_any('data', name=name)


def save_npy_file(content, name):
    tl.files.save_any_to_npy(content, name)


def save_large_npy_file(content: np.ndarray, path: str, file_name_prefix: str, real: bool, size_for_each_file=500):
    if not os.path.exists(path):
        os.mkdir(path)
    count = content.shape[0] // size_for_each_file
    for i in range(0, count + 1):
        X = content[i * size_for_each_file: (i + 1) * size_for_each_file] if i is not count \
            else content[i * size_for_each_file:]
        y = np.zeros(X.shape[0]) if real else np.ones(X.shape[0])
        dictionary = {'X': X, 'y': y}
        file_name = file_name_prefix + '_%d' % i
        save_npy_file(dictionary, os.path.join(path, file_name))


def main():
    crop, box = crop_face_with_box(cv2.imread(os.path.join(image_path, 'image_fake_2.jpg')))
    cv2.imshow('ImageWindow', crop)
    cv2.waitKey()
    return 0


if __name__ == '__main__':
    main()
