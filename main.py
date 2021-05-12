import sys
from Visual.final import recognize
from SONAR.audio import SONAR

import numpy as np
import matplotlib.pyplot as plt
import threading

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

# create concurrent threads for each object
threads = []
# camera thread
threads.append(ASLThread(1, lambda: recognize(s.abort)))
# transmitter thread
threads.append(ASLThread(2, lambda: s.play_freq(440, 10)))
# receiver thread
#threads.append(ASLThread(3, s.receive))

for thread in threads:
    thread.start()

s.receive()
plt.show()

for thread in threads:
    thread.join()

# run cleanup
s.destruct()
