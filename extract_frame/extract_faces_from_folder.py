import os
import sys
import cv2
import numpy as np
import tensorlayer as tl
from extract_frame.extract_face_from_video import crop_face


lfw_path = '/Users/twotalltotems/Documents/openface/data/lfw/raw'

file_path = os.path.dirname(os.path.realpath(__file__))
sys_path = sys.path[0]

folders = list(os.walk(lfw_path))[1:]


def show_images():
    for folder in folders:
        sub_root = folder
        # print(sub_root[0])
        for image_name in sub_root[2]:
            image_path = sub_root[0] + '/' + image_name
            cv2.imshow('img', cv2.imread(image_path))



def extract_face_from_image():
    # collection of cropped faces
    faces = []

    count = 1
    for folder in folders:
        sub_root = folder
        # print(sub_root[0])
        for image_name in sub_root[2]:
            image_path = sub_root[0] + '/' + image_name
            image = cv2.imread(image_path)
            cropped = crop_face(image)
            if cropped is not None:
                faces.append(cropped)

        print("done " + str(count))
        count += 1

    ndarray = np.asarray(faces)
    length = ndarray.shape[0]
    labels = np.zeros(length)
    dictionary = {'real_faces': ndarray, 'labels': labels}
    tl.files.save_any_to_npy(dictionary, 'dataset/real.npy')


def main():
    # extract_face_from_image()
    show_images()


if __name__ == '__main__':
    main()
    # dictionary = tl.files.load_npy_to_any('dataset', 'real.npy')
    # print(dictionary['real_faces'].shape)
    # print('finished main function')
