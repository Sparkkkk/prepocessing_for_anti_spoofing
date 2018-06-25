import tensorlayer as tl
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

data_path = os.path.join(sys.path[0], '..', 'data')
temp_data_path = os.path.join(data_path, 'temp_data')
dataset_path = os.path.join(data_path, 'dataset')
clean_data_path = os.path.join(data_path, 'clean_data')
new_dataset_path = os.path.join(data_path, 'anti_spoofing_video_dataset', 'npy')

def save_fake_face_npz():
    v0 = tl.files.load_npy_to_any(temp_data_path, 'video0.npy')
    v1 = tl.files.load_npy_to_any(temp_data_path, 'video1.npy')
    v2 = tl.files.load_npy_to_any(temp_data_path, 'video2.npy')
    v3 = tl.files.load_npy_to_any(temp_data_path, 'video3.npy')

    v = np.append(v0, v1, 0)
    v = np.append(v, v2, 0)
    v = np.append(v, v3, 0)

    length = v.shape[0]
    labels = np.ones(length)
    dictionary = {'fake_faces': v, 'labels': labels}
    tl.files.save_any_to_npy(dictionary, os.path.join(dataset_path, 'fake.npy'))


def save_dataset():
    print('loading data')
    print(dataset_path)
    real = tl.files.load_npy_to_any(clean_data_path, 'real.npy')
    fake = tl.files.load_npy_to_any(clean_data_path, 'fake.npy')
    mine = tl.files.load_npy_to_any(dataset_path, 'employee.npy')
    video4 = tl.files.load_npy_to_any(temp_data_path, 'video4.npy')
    video5 = tl.files.load_npy_to_any(temp_data_path, 'video5.npy')
    video6 = tl.files.load_npy_to_any(temp_data_path, 'video6.npy')
    video7 = tl.files.load_npy_to_any(temp_data_path, 'video7.npy')
    X = np.append(real['real_faces'], fake['fake_faces'], 0)
    y = np.append(real['labels'], fake['labels'], 0)
    X = np.append(X, mine['X'], axis=0)
    y = np.append(y, mine['y'], axis=0)
    X = np.append(X, video4['X'], axis=0)
    X = np.append(X, video5['X'], axis=0)
    X = np.append(X, video6['X'], axis=0)
    X = np.append(X, video7['X'], axis=0)
    y = np.append(y, video4['y'], axis=0)
    y = np.append(y, video5['y'], axis=0)
    y = np.append(y, video6['y'], axis=0)
    y = np.append(y, video7['y'], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.3)
    dataset = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }
    print('saving dataset')
    tl.files.save_any_to_npy(dataset, os.path.join(dataset_path, 'dataset_new.npy'))


def load_dataset():
    X = np.ndarray([0, 8, 128, 128, 3])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'G_training_0.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'G_training_1.npy')['X'])
    count = X.shape[0]
    y = np.zeros(count)
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_0.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_1.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_2.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_3.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_4.npy')['X'])
    X = np.append(X, tl.files.load_npy_to_any(new_dataset_path, 'F_training_5.npy')['X'])
    count_diff = X.shape[0] - count
    y = np.append(y, np.ones(count_diff))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.3)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == '__main__':
    print(sys.path[0])


