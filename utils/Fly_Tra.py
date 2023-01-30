'''
Karobben
Chain behaviors detect
'''

import pandas as pd
import numpy as np
import math
import operator

class fly_align():

    def dist_f(self, F1, F2):
        F1 = list(F1)
        F2 = list(F2)
        Dist = math.sqrt((F1[0]-F2[0])**2 + (F1[1]- F2[1])**2)
        return Dist

    def TB_dic(self, TB, frame):
        TB[0] = pd.to_numeric(TB[0])
        TB[1] = pd.to_numeric(TB[1])
        TB[2] = pd.to_numeric(TB[2])
        TB[3] = pd.to_numeric(TB[3])
        TB[4] = pd.to_numeric(TB[4])
        fly_bodies = TB[TB[0]==0]
        fly_bodies = fly_bodies.sort_values(by=[1,2,3,4])
        fly_head   = TB[TB[0]==1]
        fly_chasing   = TB[TB[0]==3]
        fly_sing   = TB[TB[0]==4]
        fly_mating   = TB[TB[0]==5]
        FLY_matrix = {frame:{"fly_"+str(i):{"body":list(fly_bodies.iloc[i,1:])} for i in range(len(fly_bodies))}}
        return FLY_matrix

    def nearst_match(self, id_new, FLY_1, FLY_matrix, frame, Threads, MATCH=100000):
        try:
            for id_old in list(FLY_matrix[frame-1].keys()):
                FLY_2 = FLY_matrix[frame-1][id_old]["body"]
                DIST = self.dist_f(FLY_1, FLY_2)
                if DIST == 0:
                    MATCH = DIST
                    #print("Match Result:", {id_old:id_new}, MATCH)
                    return {id_old:id_new}
                elif DIST < MATCH:
                    MATCH = DIST
                    match_id = id_old
            #print("Match result:", {match_id: id_new}, MATCH)
            return {match_id: id_new}
        except:
            print("din't match, value lost")
            return {}

    def align_BS(self, FLY_matrix, FLY_matrix_tpm, frame, Threads= 0.01):
        Lost_list = False
        MATCH_result = {}
        for id_new in list(FLY_matrix_tpm[frame].keys()):
            FLY_1 = FLY_matrix_tpm[frame][id_new]["body"]
            Dic_tmp = self.nearst_match(id_new, FLY_1, FLY_matrix, frame, Threads)
            MATCH_result.update(Dic_tmp)

        '''
        Check the duplication of the result
        '''
        #print(FLY_matrix)
        if len(set(MATCH_result.values())) == len(FLY_matrix[frame-1]):
            return MATCH_result, Lost_list

        '''
        remove dupication
        '''
        ## When the result has duplicate and only one duplicate
        if (len(MATCH_result.values()) != len(set(MATCH_result.values()))) :
            AA = list(MATCH_result.values())
            ## Find the duplicate
            [AA.remove(i) for i in list(set(MATCH_result.values()))]
            ## Find the index of the duplicate
            INDEX = [i for i, x in enumerate(MATCH_result.values()) if x == AA[0]]
            ## caculate the nearset duplicate
            Compare_dic = {}
            for i in INDEX:
                new_id=list(MATCH_result.keys())[i]
                AA = self.dist_f(FLY_matrix_tpm[frame][new_id]["body"], FLY_matrix[frame-1][MATCH_result[new_id]]["body"])
                Compare_dic.update({i:AA})
            Compare_dic

            most_ID = 1
            for i in range(len(Compare_dic.keys()) -1):
                ID_1 = list(Compare_dic.keys())[i]
                ID_2 = list(Compare_dic.keys())[i+1]
                if Compare_dic[ID_1] < Compare_dic[ID_2] and Compare_dic[ID_1] < most_ID:
                    most_ID = ID_1
                elif Compare_dic[ID_2] < Compare_dic[ID_1] and Compare_dic[ID_1] < most_ID:
                    most_ID = ID_2

            # keep the nearst point
            INDEX.remove(most_ID)
            for i in INDEX:
                MATCH_result.pop(list(MATCH_result.keys())[i])
            #print("No duplicate result", MATCH_result)
            if len(MATCH_result)==len(FLY_matrix[1]):
                print("We good")
                return MATCH_result

            #print(len(FLY_matrix_tpm[frame].keys()), len(FLY_matrix[frame-1].keys()))
            if len(FLY_matrix_tpm[frame].keys()) < len(FLY_matrix[frame-1].keys()):
                print("target lost")
                #print("why not", MATCH_result)
        '''
        Check the number of the target
        '''
        if len(FLY_matrix[frame-1]) == len(MATCH_result):
            print("we good, at here")
            return MATCH_result, Lost_list
        else:
            '''
            find the lost one
            '''
            Fly_list = list(FLY_matrix[frame-1].keys())

            for i in list(set(MATCH_result.values())):
                try:
                    Fly_list.remove(i)
                except:
                    Fly_list
            #print("we lost:", Fly_list)
            '''
            try match the nearst for it/them
            '''
            Lost_dic = {}
            for id_old in Fly_list:
                MATCH = 10
                FLY_1 = FLY_matrix_tpm[frame][id_new]["body"]
                Dic_tmp = self.nearst_match(id_old, FLY_1, FLY_matrix_tpm, frame, Threads)
                Lost_dic.update(Dic_tmp)
            Lost_dic = {v: k for k, v in Lost_dic.items()}
            #print(Lost_dic)
            if len(Lost_dic) != 0:
                MATCH_result.update(Lost_dic)
            '''
            Incase there have no math at all
            '''
            Lost_list = list(FLY_matrix[frame-1].keys())
            #print(list(set(MATCH_result.keys())))
            #print(Lost_list)
            [Lost_list.remove(i) for i in list(set(MATCH_result.keys()))]
            print("\n\nLost_list\n\n", Lost_list)
            return MATCH_result, Lost_list

        #print(len(fly_bodies))


        #FLY_matrix

    def align(self, FLY_matrix, FLY_matrix_tpm, frame, Threads= 0.3):
        Lost_list = False
        MATCH_result = {}
        for id_new in FLY_matrix_tpm[frame]:
            for id_old in FLY_matrix[frame -1 ]:
                #print(FLY_matrix[frame-1][id_old]["body"])
                #print(FLY_matrix_tpm[frame][id_new]["body"])
                Dist = self.dist_f(FLY_matrix[frame-1][id_old]["body"], FLY_matrix_tpm[frame][id_new]["body"])
                MATCH_result.update({id_old+"vs"+id_new: Dist} )
        #MATCH_result = {k: v for k, v in sorted(MATCH_result.values())}
        MATCH_result = dict( sorted(MATCH_result.items(), key=operator.itemgetter(1),reverse=False))
        RESULT_old = [list(MATCH_result.keys())[0].split("vs")[0]]
        RESULT_new = [list(MATCH_result.keys())[0].split("vs")[1]]
        MATCH_result.pop(list(MATCH_result.keys())[0])
        #print(MATCH_result)
        while len(MATCH_result)> 0:
            tmp_old = list(MATCH_result.keys())[0].split("vs")[0]
            tmp_new = list(MATCH_result.keys())[0].split("vs")[1]
            if tmp_old not in RESULT_old and tmp_new not in RESULT_new and MATCH_result[tmp_old+"vs"+tmp_new]< Threads:
                RESULT_old += [tmp_old]
                RESULT_new += [tmp_new]
            MATCH_result.pop(tmp_old+"vs"+tmp_new)
        MATCH_result = {old:new for old, new in zip(RESULT_old, RESULT_new)}
        return MATCH_result
    ## Target lost, reassign the ID after a specific of time.
