import cv2
import os
import numpy as np
import tensorlayer as tl
from core import Detector
from core import Alignment
from src.utility import min_max_regions

# set path
file_path = os.path.dirname(os.path.realpath(__file__))
frames_path = os.path.join(file_path, 'frames')
local_video_path = os.path.join(file_path, '..', 'data', 'videos')
dataset_path = os.path.join(file_path, '..', 'data', 'dataset')

if not os.path.exists(frames_path):
    os.mkdir(frames_path)

# size of cropped face
size = '128, 128'
detector = Detector()
aligner = Alignment()


def crop_face(frame):
    box = detector.detect_face(frame)
    if box is None:
        return
    face_ndarray = aligner.warp_image(frame, box[0, 0:4], image_size=size)
    # face_ndarray = cv2.cvtColor(face_ndarray, cv2.COLOR_BGR2RGB)
    # face_ndarray = np.transpose(face_ndarray, (2, 0, 1))
    return face_ndarray


def extract_frames_from_video(video_name, dataset, jump_frame, video_path):
    dictionary = extract_face(jump_frame, video_name, video_path)
    # tl.files.save_any_to_npy(dictionary, os.path.join(dataset, '%s.npy' % video_name))
    save_npy_file(dictionary, os.path.join(dataset, '%s.npy' % video_name))


def extract_face(video_name, jump_frame, video_path):
    cap = cv2.VideoCapture(os.path.join(video_path, '%s.mp4' % video_name))
    count_stored_images = 0
    count_frames = 0
    faces = []
    while cap.isOpened():
        count_frames += 1

        ret, frame = cap.read()

        if not count_frames % jump_frame == jump_frame // 2:
            continue

        if ret is True:
            print('Read %d frame: ' % count_stored_images, ret)
            cropped = crop_face(frame)
            if cropped is not None:
                cv2.imwrite(os.path.join(frames_path, 'frame_%d.jpg' % count_stored_images), cropped)
                faces.append(cropped)
                count_stored_images += 1
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    ndarray = np.asarray(faces)
    length = ndarray.shape[0]
    labels = np.ones(length)
    dictionary = {'X': ndarray, 'y': labels}
    return dictionary


def extract_faces_for_3d_cnn(n_frames, video_name, jump_frame, video_path):
    cap = cv2.VideoCapture(os.path.join(video_path, '%s.mp4' % video_name))
    count_frames = 0
    frames = []
    regions = []
    X = []
    while cap.isOpened():
        count_frames += 1

        ret, single_frame = cap.read()

        if not count_frames % jump_frame == jump_frame // 2:
            continue

        if ret is True:
            if len(frames) == n_frames:
                region = min_max_regions(np.asarray(regions))
                x = np.asarray([aligner.warp_image(frame, region, image_size=size) for frame in frames])
                X.append(x)
                frames = []
                regions = []
            else:
                boxes = detector.detect_face(single_frame)
                if boxes is not None:
                    frames.append(single_frame)
                    box = boxes[0, 0:4]
                    regions.append(box)
        else:
            break
    X = np.asarray(X)
    # print(X.shape)
    return X


def read_npy_file(name):
    return tl.files.load_npy_to_any('data', name=name)


def save_npy_file(content, name):
    tl.files.save_any_to_npy(content, name)


def main():
    extract_frames_from_video('video7', dataset_path, 15, local_video_path)


if __name__ == '__main__':
    main()
    # fake_y = np.zeros(1925)
    # save_npy_file('fake_faces_y.npy', fake_y)
    # print(read_npy_file('fake_faces_y.npy').shape)


