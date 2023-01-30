#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd



class Ink:
    def Reduice(self, img):
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

    def Frame_monochrom(self, frame):
        frame = cv2.resize(frame, (480,360))
        frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        (thresh, frame) = cv2.threshold(frame_grey, 50, 255, cv2.THRESH_BINARY)
        return frame

    def Frame_monochrom_2(self, frame):
        frame = cv2.resize(frame, (480,360))
        frame_grey = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        (thresh, frame) = cv2.threshold(frame_grey, 50, 255, cv2.THRESH_BINARY)
        return frame

    def Moving_idex_cal(self, frame1, frame2):
        frame1 = self.Frame_monochrom_2(frame1)
        frame2 = self.Frame_monochrom_2(frame2)
        Result= 100 * (frame2 - frame1).sum()/frame1.sum()
        return Result

    def __init__(self, frame_tmp, frame, Num, Result, Result_mv_index):
        #self.mv_index =
        ## Convert the frame to grey
        blackAndWhiteImage = self.Frame_monochrom(frame)
        tmp  = blackAndWhiteImage
        tmp[blackAndWhiteImage==0] = 1
        tmp[blackAndWhiteImage==255] = 0
        rmp = np.array(tmp, dtype="uint16")
        Result += tmp

        '''
        Moving index calculate
        '''
        mv_index = self.Moving_idex_cal(frame_tmp, frame)
        Result_mv_index += [[Num,mv_index]]
        self.Result = Result
        self.frame_tmp = frame
        self.mv_index = Result_mv_index
