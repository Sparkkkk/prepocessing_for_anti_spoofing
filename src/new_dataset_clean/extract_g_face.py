import os
import sys
import numpy as np
from src.extract_face_from_video import extract_faces_for_3d_cnn
from src import save_large_npy_file

file_path = sys.path[0]
dataset = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'data', 'anti_spoofing_video_dataset')
training = os.path.join(dataset, 'training')
testing = os.path.join(dataset, 'testing')


def extract_face(training: bool, real: bool, number_frame_per_object=8, jump_frames=7) -> np.ndarray:
    path = os.path.join(dataset, 'training') if training else os.path.join(dataset, 'testing')
    X = np.ndarray([0, number_frame_per_object, 128, 128, 3])
    folders_name = list(os.walk(path))[1:]
    for folder in folders_name:
        print(folder)
        for video in folder[2]:
            if video.endswith('mp4'):
                video_path = folder[0]
                video_name = video.replace('.mp4', '')
                prefix_condition = video_name.startswith('G') if real else not video_name.startswith('G')
                if prefix_condition:
                    x = extract_faces_for_3d_cnn(number_frame_per_object, video_name, jump_frames, video_path)
                    if x.shape == (x.shape[0], number_frame_per_object, 128, 128, 3):
                        X = np.append(X, x, axis=0)
                        print('finished %s.mp4' % video_name)
                    else:
                        print('failed to append %s.mp4' % video_name)
    print(X.shape)
    return X


def main():
    content = extract_face(True, False)
    path = os.path.join(dataset, 'npy')
    save_large_npy_file(content, path, 'F_training', False, 500)


if __name__ == '__main__':
    main()


