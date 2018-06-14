import cv2
from core import Detector
from core import Alignment
from core import Encoder
from src.utility import crop_face_with_box
from src.evaluate_model import evaluate_model
import tensorlayer as tl
import numpy as np
import time


detector = Detector()
aligner = Alignment()
encoder = Encoder()


class Webcam():

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # self.faces = self.load_faces(['Vinson', 'Spark', 'Zhen'])
        self.video.set(3, 1920)
        self.video.set(4, 1080)
        self.session, self.network, self.x, self.y_op = evaluate_model(None, None,
                                                                       model_name='LeNet_anti_spoofing_989_with_my_img.npz',
                                                                       test_accurary=False)

    def run(self):
        while self.video.isOpened():
            ret, frame = self.video.read()

            cropped, boxes = crop_face_with_box(frame)
            label_text = 'processing'
            if cropped is not None and boxes is not None:
                X = np.asarray([cropped])
                label = tl.utils.predict(self.session, self.network, X, self.x, self.y_op)[0]
                if label == 0:
                    label_text = 'real'
                    print(label_text)
                else:
                    label_text = 'fake'
                    print(label_text)
                box = boxes[0]
                # Draw a box around the face
                left = int(box[0])
                bottom = int(box[1])
                right = int(box[2])
                top = int(box[3])
                cv2.rectangle(frame, (left, bottom), (right, top), (0, 0, 255), 2)
                # Draw a label with a name below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, label_text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Webcam().run()


