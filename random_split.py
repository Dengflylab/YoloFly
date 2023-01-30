#!/usr/bin/env python3

import os
import random
from shutil import copyfile


INPUT = "../png_DB/png_argument"
TARGET = "../png_DB/fly2"
List  = os.popen("ls " + INPUT +"/*.png| sed 's/png$//'").read().split("\n")[:-1]

N_Toltal = len(List)
N_Train  = int(len(List)* 0.70)
N_Test  = int(len(List)* 0.1)
N_Value  = len(List) - N_Train - N_Test

F_test = random.sample(List, N_Test)
List_tmp = [list_left for list_left in List if list_left not in F_test]
F_valid = random.sample(List_tmp, N_Value)
F_train =  [ Train for Train in List_tmp if Train not in F_valid]

os.system("rm "+ TARGET + "/*/*/*")

for i in F_train:
    copyfile(i+"png", TARGET +"/train/images/" + i.split("/")[-1]+"png")
    copyfile(i+"txt", TARGET +"/train/labels/" + i.split("/")[-1]+"txt")

for i in F_valid:
    copyfile(i+"png", TARGET +"/valid/images/" + i.split("/")[-1]+"png")
    copyfile(i+"txt", TARGET +"/valid/labels/" + i.split("/")[-1]+"txt")

for i in F_test:
    copyfile(i+"png", TARGET +"/test/images/" + i.split("/")[-1]+"png")
    copyfile(i+"txt", TARGET +"/test/labels/" + i.split("/")[-1]+"txt")

'''
os.system("cp "+ " ".join([X+"png" for X in F_train]) + " " + TARGET +"/train/images/")
os.system("cp "+ " ".join([X+"txt" for X in F_train]) + " " + TARGET +"/train/labels/")
os.system("cp "+ " ".join([X+"png" for X in F_valid]) + " " + TARGET +"/valid/images/")
os.system("cp "+ " ".join([X+"txt" for X in F_valid]) + " " + TARGET +"/valid/labels/")
os.system("cp "+ " ".join([X+"png" for X in F_test]) + " " + TARGET +"/test/images/")
os.system("cp "+ " ".join([X+"txt" for X in F_test]) + " " + TARGET +"/test/labels/")
'''
