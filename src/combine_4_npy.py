import tensorlayer as tl
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os

data_path = os.path.join(sys.path[0], '..', 'data')
temp_data_path = os.path.join(data_path, 'temp_data')
dataset_path = os.path.join(data_path, 'dataset')


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
tl.files.save_any_to_npy(dictionary, 'dataset/fake.npy')

print('loading data')

real = tl.files.load_npy_to_any(dataset_path, 'real.npy')
fake = tl.files.load_npy_to_any(dataset_path, 'fake.npy')

X = np.append(real['real_faces'], fake['fake_faces'], 0)
y = np.append(real['labels'], fake['labels'], 0)

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

# tl.files.save_any_to_npy(dataset, 'dataset/dataset.npy')






