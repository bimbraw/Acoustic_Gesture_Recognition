#Author - Keshav Bimbraw

import numpy as np
import cv2
import pyautogui
from screeninfo import get_monitors
import time
import winsound

#With a 0.1 s sleep - the rate of data collection
#was around 5 Hz. For no sleep - around 14 Hz
time_start = time.perf_counter()

# frequency is set to 500Hz
freq = [587, 622, 659, 699, 734, 784, 831, 880, 932, 988, 1047, 1109, 1175, 1245]
test_list = [int(i) for i in freq]

# duration is set to 100 milliseconds
dur = 200

rounds = 5
classes = 12
len_classes = 100
configurations = ["perp", "mirror"]
configuration = configurations[0]
subjects = ["Subject_1", "Subject_2", "Subject_3"]
subject = subjects[1]

winsound.Beep(freq[-1], 200)
p = 0
q = 0
a = 1
for j in range(1, rounds+1):
    for i in range(0, classes*len_classes):
        #1920x1080
        image = pyautogui.screenshot(region=(635, 55, 640, 640))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        if i < 10:
            cv2.imwrite("B:/PhD/Ultrasound_Mirror_Journal/Data/" + str(configuration) + "/" + str(subject) + "/Raw_Data/image_" + str(j) + "_000" + str(i) + ".png", image)
        elif i >= 10 and i < 100:
            cv2.imwrite("B:/PhD/Ultrasound_Mirror_Journal/Data/" + str(configuration) + "/" + str(subject) + "/Raw_Data/image_" + str(j) + "_00" + str(i) + ".png", image)
        elif i >= 100 and i < 1000:
            cv2.imwrite("B:/PhD/Ultrasound_Mirror_Journal/Data/" + str(configuration) + "/" + str(subject) + "/Raw_Data/image_" + str(j) + "_0" + str(i) + ".png", image)
        else:
            cv2.imwrite("B:/PhD/Ultrasound_Mirror_Journal/Data/" + str(configuration) + "/" + str(subject) + "/Raw_Data/image_" + str(j) + "_" + str(i) + ".png", image)
        #top left corner to bottom right corner
        if q < 100:
            print(j, p)
            for pp in range(0, 100):
                for qq in range(0, 100):
                    a = np.power((pp+1), (qq+1))
        else:
            winsound.Beep(freq[0], 100)
            winsound.Beep(freq[6], 100)
            winsound.Beep(freq[12], 100)
            print("Move to the next gesture!")
            time.sleep(2)
            winsound.Beep(freq[p], dur)
            p = p + 1
            q = 0

        q = q + 1
    winsound.Beep(freq[0], 100)
    winsound.Beep(freq[3], 100)
    winsound.Beep(freq[6], 100)
    winsound.Beep(freq[9], 100)
    winsound.Beep(freq[12], 100)
    print("Move to the next round!")
    time.sleep(5)
    winsound.Beep(freq[-1], 200)
    p = 0
    q = 0
    a = 1

time_end = time.perf_counter()
print(time_end)

# frame_rate = ((classes*len_classes*rounds) - (rounds * 300 * classes))/time_end
text_file = open("B:/PhD/Ultrasound_Mirror_Journal/Data/" + str(configuration) + "/" + str(subject) + "/Raw_Data/frame_rate.txt", "w")
# text_file.write(str(frame_rate) + " Hz\n")
text_file.write("Total Time: "+ str(time_end-time_start)+ " s\n")
text_file.write(str(rounds) + " rounds, " + str(classes) + " classes with a length of " + str(len_classes) + " per class for configuration: " + str(configuration))
text_file.close()

# ~ 2 - 2.5 minutes per round
# 10 rounds: 25 - 30 minutes
# 5 rounds: max 15 minutes. Mirror + perpendicular - total less than 35 minutes