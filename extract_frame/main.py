from extract_frame.extract_face_from_video import crop_face, extract_frames
import tensorlayer as tl
import numpy as np
from extract_frame.evaluate_model import evaluate_model
import sys
import os
import cv2

path = sys.path[0]
model_path = os.path.join(path, '..', 'model')
data_path = os.path.join(path, '..', 'data')
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(data_path, 'image')


def show_image(img):
    cv2.imshow(' ', img)


if __name__ == '__main__':
    extract_frames(video_name='fake_face')
    evaluate_model(model_name='LeNet_anti_spoofing_99.npz')
