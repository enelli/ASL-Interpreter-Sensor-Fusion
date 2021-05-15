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

COUNT = 300
THRESHOLD = 0

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

        index = np.argmax(y, axis=1)
        confidence = y[0][index][0]

        print(index_to_letter[int(index)])

        if confidence > THRESHOLD:
            current_letter = index_to_letter[int(index)]
            buffer.append((current_letter, confidence))
            buffer.pop(0)

        average_confidences = {}

        for data_item in buffer:
            letter = data_item[0]

            if letter in average_confidences:
                average_confidences[letter] = average_confidences[letter][0] + data_item[1], average_confidences[letter][1] + 1
            else:
                average_confidences[letter] = data_item[1], 1
        
        print(average_confidences)
        best_confidence = 0
        best_letter = None

        for letter in average_confidences:
            sum_confidence = average_confidences[letter][0]
            count_confidence = average_confidences[letter][1]

            if (count_confidence > COUNT/len(average_confidences)):
                average_confidence = sum_confidence/count_confidence + count_confidence/COUNT
                if average_confidence > best_confidence:
                    best_letter = letter
                    best_confidence = average_confidence
                    print(best_letter, best_confidence)

        letter = best_letter

        if (previous_letter == 'I' and sonar.is_moving()):
            letter = 'J'

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, index_to_letter[int(index)], (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, "Moving: " + str(sonar.movement_detected), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        previous_letter = letter

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sonar.destruct()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_signs(SONAR())
