import cv2
from cv2 import data
import numpy as np
import onnxruntime as ort
import time
import os
import sys

path = os.path.dirname(os.path.realpath(__file__))
src = "/".join(path.split('/')[:-1])

sys.path.append(path)
sys.path.append(src) 

from src.SONAR.audio import SONAR

COUNT = 150
THRESHOLD = 10
CONFIDENCE_THRESHOLD = 1

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]


def detect_signs(sonar):
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.
    buffer = [(None, 0) for _ in range(COUNT)]
    previous_letter = None

    # create runnable session with exported model
    ort_session = ort.InferenceSession(path + "/signlanguage.onnx")
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # preprocess data
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]      

        y[0][8] += 10

        index = np.argmax(y, axis=1)
        confidence = y[0][index][0]
        current_letter = None

        if sonar.movement_flag and previous_letter == 'J':
            previous_letter = None

        # propagate buffer
        if confidence > THRESHOLD and not sonar.is_moving():
            current_letter = index_to_letter[int(index)]
            buffer.append((current_letter, confidence))
            buffer.pop(0)


        # find average confidences per letter
        average_confidences = {}

        for data_item in buffer:
            letter = data_item[0]

            if letter in average_confidences:
                average_confidences[letter] = average_confidences[letter][0] + data_item[1], average_confidences[letter][1] + 1
            else:
                average_confidences[letter] = data_item[1], 1

        # find most confident letter
        best_confidence = CONFIDENCE_THRESHOLD
        best_letter = None

        for letter in average_confidences:
            sum_confidence = average_confidences[letter][0]
            count_confidence = average_confidences[letter][1]

            if (count_confidence > COUNT/len(average_confidences)):
                # weight average confidence by frequency and average of confidences in buffer
                average_confidence = sum_confidence/count_confidence + 3 * count_confidence/COUNT
                if average_confidence > best_confidence:
                    best_letter = letter
                    best_confidence = average_confidence

        if previous_letter != 'J':
            letter = best_letter
        else:
            letter = 'J'

        print(letter, previous_letter, sonar.movement_flag)
        if (previous_letter == 'I' and sonar.movement_flag):
            letter = 'J'

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, current_letter, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        if previous_letter != letter:
            previous_letter = letter

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sonar.destruct()
            break

        sonar.movement_flag = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_signs(SONAR())
