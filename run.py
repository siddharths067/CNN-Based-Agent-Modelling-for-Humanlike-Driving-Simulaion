import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import *
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D

import DirectInputRoutines as DIR
from PIL import ImageGrab

from keras import optimizers
from keras import backend as K
from keras.utils import *
import pyautogui as pg
import cv2

# Global variable
OUT_SHAPE = 6
INPUT_SHAPE = (800, 600,1)


def auto_canny(image, sigma=0.9):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
def create_model(keep_prob = 0.8):
    model = Sequential()

    # NVIDIA's model
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape= INPUT_SHAPE))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(110, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign'))

    return model

pg.hotkey("alt","tab")
actor = create_model()
actor.load_weights("model_weights2.h5")

print("printing model")
print(actor)
while True:
    
    printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    printscreen = auto_canny(cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY))
    
    res = actor.predict(np.reshape(printscreen,(1,800,600,1)))
    res=res[0]
    print(res)
    #if(res[0] >= 0.5):
    DIR.PressKey(DIR.W)
    #time.sleep(0.20)
    #DIR.ReleaseKey(DIR.W)
    if(res[1] >= 0.5):
        DIR.PressKey(DIR.S)
        time.sleep(0.20)
        DIR.ReleaseKey(DIR.S)
    if(res[2] >= 0.95):
        DIR.PressKey(DIR.A)
        time.sleep(0.1)
        DIR.ReleaseKey(DIR.A)
    if(res[3] >= 0.98):
        DIR.PressKey(DIR.D)
        time.sleep(0.1)
        DIR.ReleaseKey(DIR.D)
    

