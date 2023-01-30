'''
Karobben
Chain behaviors detect
'''
import math
import pandas as pd
# TB example
# TB = pd.read_csv("csv/20210712-C0147-WASH_backgroud-5th-29C_6d_Trim.mp4.csv", sep=" ", header = None)
class Chain_finder:

    Chain_result = []

    def __init__(self,TB, Diamiter=.1, Chain_Num = 3, Angle = 15):
        TB = TB[TB[0]==3]
        tmp_result = []
        Flies = [[float(TB.iloc[i, 1]), float(TB.iloc[i,2]),  float(TB.iloc[i,3]),  float(TB.iloc[i,4])] for i in range(len(TB))]
        for i in Flies:
            for ii in Flies:
                if i != ii:
                    dist = self.dist_f(i, ii)
                    print("\n\n\n", dist, "\n\n")
                    if dist <=Diamiter:
                        #print(dist)
                        tmp_result += [[i, ii]]
        self.Chain_result = tmp_result


    def dist_f(self, F1, F2):
        Dist = math.sqrt((F1[0]-F2[0])**2 + (F1[1]- F2[1])**2)
        return Dist
