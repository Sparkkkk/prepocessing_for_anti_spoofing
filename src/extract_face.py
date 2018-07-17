import os
import sys
import numpy as np
from src.extract_face_from_video import extract_faces_for_3d_cnn
from src.utility import save_npy_file

file_path = sys.path[0]
dataset = os.path.join(os.path.dirname(file_path), 'data', 'anti_spoofing_video_dataset')
training = os.path.join(dataset, 'training')
testing = os.path.join(dataset, 'testing')

array_cameras = ['HW', 'HS', '5s', 'IP', 'ZTE']


def extract_face(training: bool, real: bool, number_frame_per_object=8, jump_frames=7) -> np.ndarray:
    path = os.path.join(dataset, 'training') if training else os.path.join(dataset, 'testing')
    X = np.ndarray([0, number_frame_per_object, 128, 128, 3])
    y_domain = []
    counts_for_each_file = 400
    file_prefix_0 = 'G' if real else 'F'
    file_prefix_1 = 'training' if training else 'testing'
    folders_name = list(os.walk(path))[1:]
    file_count = 0
    for folder in folders_name:
        # print(folder)
        for video in folder[2]:
            if video.endswith('mp4'):
                video_path = folder[0]
                video_name = video.replace('.mp4', '')
                prefix_condition = video_name.startswith('G') if real else not video_name.startswith('G')
                # test_condition = video_name.split('_')[5] == '6'
                if prefix_condition:
                    print('running %s.mp4' % video_name)
                    x = extract_faces_for_3d_cnn(number_frame_per_object, video_name, jump_frames, video_path)
                    if x.shape == (x.shape[0], number_frame_per_object, 128, 128, 3):
                        # X = np.append(X, x, axis=0)
                        domain = array_cameras.index(video_name.split("_")[2])
                        X = np.vstack((X, x))
                        y_domain += x.shape[0] * [domain]
                        del domain
                        del x
                        print('finished %s.mp4' % video_name)
                        if X.shape[0] >= counts_for_each_file:
                            y_domain = np.expand_dims(np.asarray(y_domain), axis=1)
                            y = np.zeros((X.shape[0], 1)) if real else np.ones((X.shape[0], 1))
                            print('y', y.shape)
                            print('y_domain', y_domain.shape)
                            y = np.concatenate([y, y_domain], axis=1)
                            dictionary = {'X': X, 'y': y}
                            save_npy_file(dictionary, os.path.join(dataset, 'npy', '%s_%s_j%d_%d' % (file_prefix_0, file_prefix_1, jump_frames, file_count)))
                            file_count += 1
                            X = np.ndarray([0, number_frame_per_object, 128, 128, 3])
                            y_domain = []
                            del y
                            print('save file %d' % file_count)
                    else:
                        print('failed to append %s.mp4' % video_name)
    print(X.shape)
    if X.shape[0] < counts_for_each_file:
        y_domain = np.expand_dims(np.asarray(y_domain), axis=1)
        y = np.zeros((X.shape[0], 1)) if real else np.ones((X.shape[0], 1))
        y = np.concatenate([y, y_domain], axis=1)
        dictionary = {'X': X, 'y': y}
        save_npy_file(dictionary, os.path.join(dataset, 'npy', '%s_%s_j%d_%d' % (file_prefix_0, file_prefix_1, jump_frames, file_count)))
    return X


def main():
    extract_face(training=True, real=False, jump_frames=1)


if __name__ == '__main__':
    main()


