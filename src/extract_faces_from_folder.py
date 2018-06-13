import os
import sys
import cv2
import numpy as np
import tensorlayer as tl
from src.extract_face_from_video import crop_face

file_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_path, '..', 'data')
image_folder_path = os.path.join(data_path, 'image')
dataset_folder_path = os.path.join(data_path, 'dataset')
lfw_path = os.path.join(image_folder_path, 'raw')
frames_path = os.path.join(file_path, 'frames')

sys_path = sys.path[0]


def extract_face_from_image(npy_file_name, image_folder_name=None, lfw=True):
    # collection of cropped faces
    faces = []
    count_stored_images = 0
    if lfw:
        folders = list(os.walk(lfw_path))[1:]
    else:
        folders = list(os.walk(os.path.join(image_folder_path, image_folder_name)))[1:]

    count = 1
    for folder in folders:
        sub_root = folder
        # print(sub_root[0])
        for image_name in sub_root[2]:
            image_path = sub_root[0] + '/' + image_name
            image = cv2.imread(image_path)
            cropped = crop_face(image)
            if cropped is not None:
                cv2.imwrite(os.path.join(frames_path, 'frame_%d.jpg' % count_stored_images), cropped)
                faces.append(cropped)
                count_stored_images += 1

        print("done " + str(count))
        count += 1

    ndarray = np.asarray(faces)
    length = ndarray.shape[0]
    labels = np.zeros(length)
    dictionary = {'X': ndarray, 'y': labels}
    tl.files.save_any_to_npy(dictionary, os.path.join(dataset_folder_path, '%s.npy' % npy_file_name))
    print('saved! number of image is %d' % count_stored_images)


def main():
    extract_face_from_image('real')


if __name__ == '__main__':
    main()
