#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input','-i',   help='inputfile')

args = parser.parse_args()
Video = args.input

def Reduice(img):
    Result=[]
    H = len(img)
    W = len(img[0])
    for i in range(1,H-1):
        for ii in range(1,W-1):
            pix = [img[i-1][ii], # LEFT
                   img[i+1][ii], # RIGHT
                   img[i][ii-1], # UP
                   img[i][ii+1], # DOWN
                   img[i-1][ii-1], # LU
                   img[i+1][ii-1], # RU
                   img[i-1][ii+1], # LD
                   img[i+1][ii+1] # RD
                  ]
            if pix ==[0, 0, 0, 0, 0, 0, 0, 0]:
                Result +=[[i,ii]]
    return Result

def Frame_monochrom(frame):
    frame = cv2.resize(frame, (480,360))
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    (thresh, frame) = cv2.threshold(frame_grey, 50, 255, cv2.THRESH_BINARY)
    return frame

def Frame_monochrom_2(frame):
    frame = cv2.resize(frame, (480,360))
    frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    (thresh, frame) = cv2.threshold(frame_grey, 50, 255, cv2.THRESH_BINARY)
    return frame

def Moving_idex_cal(frame1, frame2):
    frame1 = Frame_monochrom_2(frame1)
    frame2 = Frame_monochrom_2(frame2)
    Result= 100 * (frame2 - frame1).sum()/frame1.sum()
    return Result




cap=cv2.VideoCapture(Video)
ret,frame_tmp=cap.read()

Result = Frame_monochrom(frame_tmp)
Result[Result==0] = 0
Result[Result==255] = 0
Result = np.array(Result, dtype="uint16")
print(Result)

Num = 0

#print(frame_tmp)

Result_mv_index = []
while (True):
    try:
        Num +=1
        ret,frame = cap.read()
        ## Convert the frame to grey
        #frame = cv2.resize(frame, (480,360))
        blackAndWhiteImage = Frame_monochrom(frame)
        '''
        black_result = Reduice(blackAndWhiteImage)
        img2 = np.array(blackAndWhiteImage)
        img2[img2!=254]=254
        for i in black_result:
            img2[i[0],i[1]]=blackAndWhiteImage[i[0],i[1]]
        print(img2, Num)
        '''
        tmp  = blackAndWhiteImage
        tmp[blackAndWhiteImage==0] = 1
        tmp[blackAndWhiteImage==255] = 0
        Result += np.array(tmp, dtype="uint16")
        img = Result
        '''
        Moving index calculate
        '''
        mv_index = Moving_idex_cal(frame_tmp, frame)
        frame_tmp = frame
        Result_mv_index += [[Num,mv_index]]
        print(mv_index)
        '''
        cv2.imshow("video", Frame_monochrom(frame))
        if cv2.waitKey(1)&0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
        '''
    except:
        break

TB = pd.DataFrame(Result)
TB.to_csv(Video+"_map.csv")
TB2 = pd.DataFrame(Result_mv_index)
TB2.to_csv(Video+"_mv_index.csv")
