import cv2
import numpy as np
import onnxruntime as ort
import time

DELAY = 500
THRESHOLD = 20
TIME_DELAY = 500

def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]


def main():
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")


    letters = [None, None]
    moving = False

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

        if y[0][index] > THRESHOLD and not moving:
            letter = index_to_letter[int(index)]
        else:
            letter = None

        if letter != letters[1]:
            letters = [letters[1], letter]
            print(letters)

        if (letters[0] == 'I' and letters[1] is None and moving):
            letter = 'J'

        
        """
        if start is None and movement:
            start = time.time(), letter
            average_confidence += 0, 0
        else:
            average_confidence += y[0][index], average_confidence[1] + 1
        
        #print(y[0][index])
        """

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.putText(frame, "Moving: " + str(moving), (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('m'):
            moving = not moving
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
