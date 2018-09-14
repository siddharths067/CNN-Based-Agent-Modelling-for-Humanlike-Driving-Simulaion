import pyautogui as pg
import time
import DirectInputRoutines as DIR

pg.hotkey("alt","tab")
for i in range(1000000):
    DIR.PressKey(DIR.W)
    time.sleep(2)
    if (i+1) % 10 == 0:
        DIR.PressKey(DIR.A)
        time.sleep(0.20)
        DIR.ReleaseKey(DIR.A)
    if (i+1) %  20 ==0:
        DIR.PressKey(DIR.D)
        time.sleep(0.20)
        DIR.ReleaseKey(DIR.D)
    if(i+1) % 25 == 0 :
        DIR.PressKey(DIR.B)
        time.sleep(6)
        DIR.ReleaseKey(DIR.B)