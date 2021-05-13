import sys
from SONAR.audio import SONAR

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threading

matplotlib.use('TkAgg')

# determine whether this device is transmitter or receiver
TRANSMITTER = (len(sys.argv) >= 2 and sys.argv[1] == '-t')
if not TRANSMITTER:
    from Visual.final import recognize

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
LOW_FREQ = 19220
HIGH_FREQ = 19880
TRANSMIT_FREQ = (LOW_FREQ + HIGH_FREQ) / 2
DURATION = 5  # seconds
s.set_freq_range(LOW_FREQ, HIGH_FREQ)

# create concurrent threads for each object
threads = []
if not TRANSMITTER:
    # camera thread
    threads.append(ASLThread(1, lambda: recognize(s.abort)))
    plt.ion()
    plt.show()
else:
    # transmitter thread
    threads.append(ASLThread(2, lambda: s.play_freq(TRANSMIT_FREQ)))

for thread in threads:
    thread.start()

if not TRANSMITTER:
    # handle receiver thread in main
    s.receive_burst()
else:
    try:
        while not s.terminate:
        # TODO: do something better than this
            pass
    except KeyboardInterrupt:
        s.abort()

for thread in threads:
    thread.join()

# run cleanup
s.destruct()
