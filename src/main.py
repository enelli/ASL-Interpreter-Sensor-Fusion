import sys
sys.path.append('/Users/ellenwang/Desktop/Sign-Language-Interpreter-using-Deep-Learning') 

from src.SONAR.audio import SONAR
from src.Visual2.step_5_camera import detect_signs

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threading

matplotlib.use('TkAgg')

# determine whether this device is transmitter or receiver
#TRANSMITTER = (len(sys.argv) >= 2 and sys.argv[1] == '-t')
#if not TRANSMITTER:
# from Visual.final import recognize

# create separate threads for video and SONAR
class ASLThread(threading.Thread):
    def __init__(self, threadID, function):
        threading.Thread.__init__(self)
        self.id = threadID
        self.func = function  # update function to run continuously

    def run(self):
        self.func()

# create audio object
s = SONAR()

# audio configuration
LOW_FREQ = 220
HIGH_FREQ = 1760
TRANSMIT_FREQ = (LOW_FREQ + HIGH_FREQ) / 2
DURATION = 5  # seconds

s.set_freq_range(220, 20000)

# create concurrent threads for each object
threads = []
# camera thread

# transmitter thread
# threads.append(ASLThread(2, lambda: s.play_freq(440)))
# receiver thread
threads.append(ASLThread(2, lambda: s.play_freq(19440)))
threads.append(ASLThread(3, s.receive_burst))

plt.ion()
plt.show()

for thread in threads:
    thread.start()

detect_signs(s)

for thread in threads:
    thread.join()

# run cleanup
s.destruct()
