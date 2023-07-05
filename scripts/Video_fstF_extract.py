#!/usr/bin/env python3
import cv2, os
import pandas as pd

TB = pd.read_excel("Video_list.xlsx")


for i in range(len(TB)):
    V_loc   = TMP[1] + "/" +TMP[2]
    cap=cv2.VideoCapture(V_loc)
    ret,frame=cap.read()
    cv2.imwrite(TMP[2] + '.png',frame)
