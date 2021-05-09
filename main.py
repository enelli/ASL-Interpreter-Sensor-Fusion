import sys
from Visual.final import recognize
from SONAR.audio import play, record

import numpy as np

WIDTH = 300
HEIGHT = 300

#play("SONAR/test.wav")

def thresh_func():
    ''' return a WIDTH x HEIGHT binary determination of 0s and 255s
    representing where the hand is, with (0,0) representing the top 
    left of the screen'''
    return np.zeros((WIDTH, HEIGHT), dtype=np.uint8)

recognize()
#recognize(thresh_func)
