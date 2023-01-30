import pandas as pd
import pandas as pd
import numpy as np
import operator

from collections import namedtuple


class head_match():

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    MATCH_result = None
    def area(self, a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy

    def Sort_uniq(self, MATCH_result):
        '''
        This function is for extract the unique match of the head. Let's say Bod A includes two head a and b, body B has only b. So, we'll give b to B and leave a to A.
        '''
        List = [i.split(":")[0] for i in MATCH_result.keys()]
        Uniq_list = []
        for i in List:
            if List.count(i) ==1:
                Uniq_list += [i]
        Uniq_dic = {}
        for i in Uniq_list:
            for Z in MATCH_result.keys():
                if i in Z:
                    Uniq_dic.update({Z:MATCH_result[Z]})

        for Z in Uniq_dic.keys():
            MATCH_result.pop(Z)
        Uniq_dic.update(MATCH_result)
        MATCH_result = Uniq_dic
        return MATCH_result
        #[i in Z for i,Z in  zip(Uniq_list, MATCH_result.keys())]

    def main(self, FLY_matrix, Num_frame, TB_head, Thread= 0.7):
        TB_head.index = range(len(TB_head.index))
        HEAD_Mdic = {}
        for fly in FLY_matrix[Num_frame].keys():
            fly_body = FLY_matrix[Num_frame][fly]['body']
            fly_loc = self.Rectangle(fly_body[0] - fly_body[2]/2, fly_body[1] - fly_body[3]/2,  fly_body[0] + fly_body[2]/2, fly_body[1] + fly_body[3]/2)
            for ID in range(len(TB_head.index)):
                head_tmp = list(TB_head.iloc[ID,1:])
                head_loc = self.Rectangle(head_tmp[0]-head_tmp[2]/2, head_tmp[1]-head_tmp[3]/2, head_tmp[0]+head_tmp[2]/2, head_tmp[1]+head_tmp[3]/2)
                R = self.area(fly_loc, head_loc)
                if R !=None:
                    R = R/(head_tmp[2]*head_tmp[3])
                    if R >= Thread:
                        HEAD_Mdic.update({":".join([fly,str(ID)]):R})

        self.MATCH_result = dict( sorted(HEAD_Mdic.items(), key=operator.itemgetter(1),reverse=True))
        self.MATCH_result = self.Sort_uniq(self.MATCH_result)

        if len(self.MATCH_result)> 0:
            RESULT_old = [list(self.MATCH_result.keys())[0].split(":")[0]]
            RESULT_new = [list(self.MATCH_result.keys())[0].split(":")[1]]
            self.MATCH_result.pop(list(self.MATCH_result.keys())[0])
            Threads = 0
            while len(self.MATCH_result)> 0:
                tmp_old = list(self.MATCH_result.keys())[0].split(":")[0]
                tmp_new = list(self.MATCH_result.keys())[0].split(":")[1]
                if tmp_old not in RESULT_old and tmp_new not in RESULT_new and self.MATCH_result[tmp_old+":"+tmp_new]> Threads:
                    RESULT_old += [tmp_old]
                    RESULT_new += [tmp_new]
                self.MATCH_result.pop(tmp_old+":"+tmp_new)
            self.MATCH_result = {old:new for old, new in zip(RESULT_old, RESULT_new)}
