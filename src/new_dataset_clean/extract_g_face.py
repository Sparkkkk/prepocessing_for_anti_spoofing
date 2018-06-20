import os
import sys
import numpy as np
from src.extract_face_from_video import extract_faces_for_3d_cnn, save_npy_file

file_path = sys.path[0]
dataset = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'data', 'anti_spoofing_video_dataset')
training = os.path.join(dataset, 'training')
testing = os.path.join(dataset, 'testing')
folders_name = list(os.walk(testing))[1:]
numbers_frame_per_object = 8
jump_frames = 7
X_real = np.ndarray([0, numbers_frame_per_object, 128, 128, 3])
X_fake = np.ndarray([0, numbers_frame_per_object, 128, 128, 3])
for folder in folders_name:
    print(folder)
    for video in folder[2]:
        if video.endswith('mp4'):
            video_path = folder[0]
            video_name = video.replace('.mp4', '')
            x = extract_faces_for_3d_cnn(numbers_frame_per_object, video_name, jump_frames, video_path)
            if x.shape == (x.shape[0], numbers_frame_per_object, 128, 128, 3):
                if video.startswith('G'):
                    X_real = np.append(X_real, x, axis=0)
                else:
                    X_fake = np.append(X_fake, x, axis=0)
                print('finished %s.mp4' % video_name)
            else:
                print('failed to append %s.mp4' % video_name,)


print(X_real.shape)
print(X_fake.shape)
y_real = np.zeros(X_real.shape[0], dtype=int)
y_fake = np.ones(X_fake.shape[0], dtype=int)
dictionary_real = {'X': X_real, 'y': y_real}
dictionary_fake = {'X': X_fake, 'y': y_fake}
save_npy_file(dictionary_real, os.path.join(dataset, 'real_faces_test.npy'))
save_npy_file(dictionary_fake, os.path.join(dataset, 'fake_faces_test.npy'))



