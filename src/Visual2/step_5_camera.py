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
DRAW_FRAMES = 20  # number of frames to hold at a J

# movement count range for j detection
J_MOVE_LOW = 7
J_MOVE_HIGH = 14

# possible letters detected at the end of a J
J_END_LETTERS = ['I', None]

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

    # track number of frames since last letter change, to allow
    # human-readable letter draw duration
    num_since_change = 0

    # create runnable session with exported model
    ort_session = ort.InferenceSession(path + "/signlanguage.onnx")
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        num_since_change += 1

        # preprocess data
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        # hold a j for at least DRAW_FRAMES
        if previous_letter == 'J' and num_since_change < DRAW_FRAMES:
            # continue showing output
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, previous_letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
            cv2.imshow("Sign Language Translator", frame)
            cv2.waitKey(1)  # ensure output is drawn
            continue

        # run the predictor
        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        # artificially bias towards 'I'
        y[0][8] += 10
        # artificially decrease 'P'
        y[0][14] -= 10

        index = np.argmax(y, axis=1)
        confidence = y[0][index][0]
        frame_letter = None

        # extract movement data from sonar
        movement_flag = sonar.is_moving()
        move_count = sonar.read_move_count()
        potential_j = J_MOVE_LOW <= move_count <= J_MOVE_HIGH

        #if move_count > 0 and previous_letter == 'J':
        #    previous_letter = None

        # propagate buffer
        if not movement_flag:
            if confidence > THRESHOLD:
                frame_letter = index_to_letter[int(index)]
                buffer.append((frame_letter, confidence))
            else:
                buffer.append((None, 1-confidence))

            buffer.pop(0)
        # frame_letter now holds the most confident letter from the current frame

        # find average confidences per letter across the buffer
        average_confidences = {}

        for data_item in buffer:
            letter = data_item[0]

            if letter in average_confidences:
                average_confidences[letter] = average_confidences[letter][0] + data_item[1], average_confidences[letter][1] + 1
            else:
                average_confidences[letter] = data_item[1], 1

        # find most confident letter across recent frames
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

        #if previous_letter != 'J':
        #    letter = best_letter
        #else:
        #    letter = 'J'
        letter = best_letter

        #print(letter, frame_letter, previous_letter, potential_j)
        if potential_j and previous_letter == 'I':
            print(frame_letter)  # to detecte J_END_LETTERS
            if frame_letter in J_END_LETTERS:
                letter = 'J'

        # mirror
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, frame_letter, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        if letter != previous_letter:
            num_since_change = 0
        previous_letter = letter

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sonar.abort()
            break

        sonar.movement_flag = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_signs(SONAR())
