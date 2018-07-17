# from src.utility import convert_pb_to_coreml, create_pb_from_ckpt
# import numpy as np
#
# if __name__ == '__main__':
#     # create_pb_from_ckpt('LeNet_anti_spoofing_97', 'Softmax')
#     model = convert_pb_to_coreml('model_new/LeNet_anti_spoofing_97.pb',
#                                  'model_new/LeNet_anti_spoofing_97.mlmodel',
#                                  {
#                                   'Placeholder:0': [100, 3380],
#                                   'Placeholder_1:0': [1, 256]},
#                                  ['Softmax:0'])
