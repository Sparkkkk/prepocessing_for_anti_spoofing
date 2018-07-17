import cv2
from core import Detector
from core import Alignment
from core import Encoder
from src.utility import crop_face_with_box
from src.evaluate_model import evaluate_model, evaluate_AsNet
from src.utility import min_max_regions
import tensorlayer as tl
import numpy as np
import time
from scipy import stats


detector = Detector()
aligner = Alignment()
encoder = Encoder()


class Webcam():

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # self.faces = self.load_faces(['Vinson', 'Spark', 'Zhen'])
        self.video.set(3, 1920)
        self.video.set(4, 1080)
        self.session, self.network, self.x, self.y_op = evaluate_AsNet(None, None,
                                                                       model_name='AsNet_c5_d1_128_300.npz',
                                                                       test_accurary=False)
        self.X = np.ndarray([0, 128, 128, 3])
        self.label_text = 'processing'
        self.count_frame = 0
        self.last_predict_time = time.time()
        self.regions = []
        self.frames = []
        self.result_queue = []
        self.n_frames = 8
        tl.files.save_any_to_npy()
        tl.files.save_npz()

    def run_3d_cnn(self):
        while self.video.isOpened():
            ret, frame = self.video.read()
            self.count_frame += 1
            cropped, boxes = crop_face_with_box(frame, size='128, 128')
            if cropped is None:
                self.label_text = 'processing'
                self.X = np.ndarray([0, 128, 128, 3])
                self.result_queue = []
            if cropped is not None and boxes is not None:
                if self.count_frame % 2 == 0:
                    x = np.asarray([cropped])
                    self.X = np.vstack((self.X, x))
                if self.X.shape[0] == 8:
                    print('ready to predict')
                    X = np.asarray([self.X])
                    label = tl.utils.predict(self.session, self.network, X, self.x, self.y_op)[0]
                    self.result_queue.insert(0, label)
                    if len(self.result_queue) >= 6:
                        self.result_queue.pop()
                    print('time difference between two prediction:', time.time() - self.last_predict_time)
                    self.last_predict_time = time.time()
                    self.X = np.ndarray([0, 128, 128, 3])
                    mode = stats.mode(self.result_queue)[0][0]
                    print(self.result_queue)
                    if mode == 0:
                        self.label_text = 'real'
                        print(self.label_text)
                    else:
                        self.label_text = 'fake'
                        print(self.label_text)
                box = boxes[0]
                # Draw a box around the face
                left = int(box[0])
                bottom = int(box[1])
                right = int(box[2])
                top = int(box[3])
                cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
                # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.label_text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

    def run_3d_cnn_0(self):
        while self.video.isOpened():
            ret, frame = self.video.read()
            boxes = detector.detect_face(frame)
            self.count_frame += 1
            if boxes is not None:
                box = boxes[0, 0:4]
                # Draw a box around the face
                left = int(box[0])
                bottom = int(box[1])
                right = int(box[2])
                top = int(box[3])
                cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
                # Draw a label with a name below the face
                if self.count_frame % 2 == 0:
                    if len(self.frames) == self.n_frames:
                        region = min_max_regions(np.asarray(self.regions))
                        x = np.asarray([aligner.warp_image(frame, region, image_size='128, 128') for frame in self.frames])
                        if x.shape != (8, 128, 128, 3):
                            continue
                        label = tl.utils.predict(self.session, self.network, np.asarray([x]), self.x, self.y_op)[0]
                        self.result_queue.insert(0, label)
                        if len(self.result_queue) >= 5:
                            self.result_queue.pop()
                        print('time difference between two prediction:', time.time() - self.last_predict_time)
                        mode = stats.mode(self.result_queue)[0][0]
                        print(self.result_queue)
                        if mode == 0:
                            self.label_text = 'real'
                            print(self.label_text)
                        else:
                            self.label_text = 'fake'
                            print(self.label_text)
                        self.last_predict_time = time.time()
                        self.frames = []
                        self.regions = []
                    else:
                        self.regions.append(box)
                        self.frames.append(frame)
                    # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.label_text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                self.label_text = 'processing'
                self.regions = []
                self.frames = []
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

    def run_2d_cnn(self):
        while self.video.isOpened():
            ret, frame = self.video.read()

            cropped, boxes = crop_face_with_box(frame)
            self.label_text = 'processing'
            if cropped is not None and boxes is not None:
                X = np.asarray([cropped])
                label = tl.utils.predict(self.session, self.network, X, self.x, self.y_op)[0]
                if label == 0:
                    self.label_text = 'real'
                    print(self.label_text)
                else:
                    self.label_text = 'fake'
                    print(self.label_text)
                box = boxes[0]
                # Draw a box around the face
                left = int(box[0])
                bottom = int(box[1])
                right = int(box[2])
                top = int(box[3])
                cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
                # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.label_text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Webcam().run_3d_cnn()


