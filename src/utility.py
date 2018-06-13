from tensorflow.python.tools.freeze_graph import freeze_graph
import tfcoreml
import numpy as np


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