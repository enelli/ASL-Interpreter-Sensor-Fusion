import sys
import os

path = os.path.dirname(os.path.realpath(__file__))
path = "/".join(path.split('/')[:-1])
sys.path.append(path) 

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
LOW_FREQ = 18000
HIGH_FREQ = 20000
TRANSMIT_FREQ = (LOW_FREQ + HIGH_FREQ) / 2

s.set_freq_range(LOW_FREQ, HIGH_FREQ)

# create concurrent threads for each object
threads = []
# camera thread

# transmitter thread
# threads.append(ASLThread(2, lambda: s.play_freq(440)))
# receiver thread
threads.append(ASLThread(2, lambda: s.play_freq(TRANSMIT_FREQ)))
threads.append(ASLThread(3, lambda: detect_signs(s)))

plt.ion()
plt.show()

if s.calibrate_thresholds(TRANSMIT_FREQ):
    for thread in threads:
        thread.start()

    s.receive_burst()

    for thread in threads:
        thread.join()

# run cleanup
s.destruct()
