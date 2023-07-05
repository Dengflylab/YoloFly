#!/usr/bin/env python3
import cv2, os
import pandas as pd

TB = pd.read_excel("Video_list.xlsx")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


for i in range(len(TB)):
    TMP     = TB.iloc[i,:]
    V_name  = TMP[2]
    V_loc   = TMP[1] + "/" +TMP[2]
    F_no    = TMP[0]
    if os.path.exists(V_loc):
        cap=cv2.VideoCapture(V_loc)
        frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_c = cap.get(cv2.CAP_PROP_FPS)
        print("Video location:", TMP[1])
        print("Video:", V_name, "in Group:", TMP[3])
        print("Number of flys:", F_no, "Total Frame of Video:", frame_total)
        if fps_c != 30:
            print(f"{bcolors.WARNING}WARNING_001: fps is not 30\nWe strogly suggests that Adjust you fps to 30. Current fps: {fps_c}{bcolors.ENDC}")


    else:
        print(f"{bcolors.FAIL}ERROR_001: Video not found\n{V_name} Not Found, please Check your location in {V_loc};\n\nPlease check your inf in excel 'Video_list.xlsx' {bcolors.ENDC}")
        raise "Error"

Video_list1 = pd.DataFrame(index= range(len(TB)))

Video_list1['Number'] = TB.iloc[:,0:1]
Video_list1['loc'] = TB.iloc[:,1]+ "/"  + TB.iloc[:,2]
Video_list1 = Video_list1[Video_list1.duplicated()==False]
Video_list1.to_csv("Video_list", sep= "\t", index=False, header=False)
TB.iloc[:,[2,4,5,6,7]].to_csv("Video_list.csv", sep= "\t", index=False, header=False)
