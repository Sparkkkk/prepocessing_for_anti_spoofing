from src.extract_face_from_video import crop_face, extract_frames_from_video
from src.extract_faces_from_folder import extract_face_from_image
import tensorlayer as tl
import numpy as np
from src.evaluate_model import evaluate_model, get_X_y_test
import sys
import os
import cv2

path = sys.path[0]
model_path = os.path.join(path, '..', 'model')
data_path = os.path.join(path, '..', 'data')
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(data_path, 'image')


def get_test_data_from_image():

    # prepare data
    img0 = cv2.imread(os.path.join(image_path, 'my_image_real.jpg'))
    img1 = cv2.imread(os.path.join(image_path, 'my_image_fake.jpg'))
    img2 = cv2.imread(os.path.join(image_path, 'image_real_1.jpg'))
    img3 = cv2.imread(os.path.join(image_path, 'image_fake_1.jpg'))
    img4 = cv2.imread(os.path.join(image_path, 'image_fake_2.jpg'))
    X_test = np.ndarray([5, 112, 112, 3])
    X_test[0] = crop_face(img0)
    X_test[1] = crop_face(img1)
    X_test[2] = crop_face(img2)
    X_test[3] = crop_face(img3)
    X_test[4] = crop_face(img4)
    y_test = np.array([0, 1, 0, 1, 1])
    return X_test, y_test


def main():
    # extract_face_from_image('zhen', 'zhen', False)
    X_test, y_test = get_X_y_test('test', 'fake_face')
    # X_test, y_test = get_test_data_from_image()
    evaluate_model(X_test, y_test, model_name='LeNet_anti_spoofing_989_with_my_img.npz')


if __name__ == '__main__':
    main()

