from src.extract_face_from_video import crop_face, extract_frames_from_video
from src.extract_faces_from_folder import extract_face_from_image
import tensorlayer as tl
import numpy as np
from src.evaluate_model import evaluate_model
import sys
import os
import cv2

path = sys.path[0]
model_path = os.path.join(path, '..', 'model')
data_path = os.path.join(path, '..', 'data')
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(data_path, 'image')

if __name__ == '__main__':
    # extract_frames_from_video(video_name='fake_face')
    # extract_face_from_image('test', 'test_faces', False)
    evaluate_model('test', model_name='LeNet_anti_spoofing_99.npz', second_file_name='fake_face')
