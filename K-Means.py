import json
import os
import numpy as np
import pandas as pd
import random
import sys
import time

from sympy import re

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
WORDLE_DATA_FILE = os.path.join(DATA_DIR, "possible_words.txt")
K_MEANS_FILE = os.path.join(DATA_DIR, "k_means.json")


class KMeansClusterer:
    def __init__(self,ndarray,cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points=self.__pick_start_point(ndarray,cluster_num)
         
    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for j in range(len(ndarray[0])):
            distance_min = sys.maxsize
            index=-1
            for i in range(len(self.points)):                
                distance = self.__center(self.points[i],j)
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index].append(j)
        
        """new_center=[]
        for item in result:
            new_center.append(self.__center(item).tolist())"""
        # 中心点未改变，说明达到稳态，结束递归
        if self.points == result:
            return result
         
        self.points=result
        return self.cluster()
             
    def __center(self,list,index):
        '''计算平均编辑距离
        '''
        # 计算每一列的平均值
        sum = 0
        for i in list:
            sum += int(ndarray[i][index])
        return sum/len(list)


    def __pick_start_point(self,ndarray,cluster_num):
        if cluster_num <0 or cluster_num > len(ndarray[0]):
            raise Exception("簇数设置有误")
        # 随机点的下标
        indexes=random.sample(np.arange(0,len(ndarray[0]),step=1).tolist(),cluster_num)
        points=[]
        for index in indexes:
            points.append([index])
        return points

def cal_dis():
    words=[]
    with open(WORDLE_DATA_FILE) as fp:
        for line in fp.readlines():
            words.append((str(line)).split("\n")[0])
    distance = [[0]*len(words) for i in range(len(words))]
    for ii in range(len(words)):
        for jj in range(len(words)):
            n = len(words[ii])+1
            m = len(words[jj])+1
            dp = [[0]*n for i in range(m)]
            #dp初始化
            dp[0][0]=0
            for i in range(1,m):
                dp[i][0] = dp[i-1][0] + 1
            for j in range(1,n):
                dp[0][j] = dp[0][j-1] + 1
            for i in range(1,m):
                for j in range(1,n):
                    if words[ii][i-1]==words[jj][j-1]:
                        temp = 0
                    else:
                        temp = 1       
                    dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+temp)
            distance[ii][jj] = dp[m-1][n-1]
    print(distance[0])
    return distance
            


if __name__ == "__main__":
    ndarray = cal_dis()
    #ndarray = [[0,2,3,5,5],[2,0,4,4,5],[3,4,0,5,2],[5,4,5,0,3],[5,5,2,3,0]]
    k1 = KMeansClusterer(ndarray,100)
    result = k1.cluster()
    words=[]
    words_dict={}
    with open(WORDLE_DATA_FILE) as fp:
        for line in fp.readlines():
            words.append((str(line)).split("\n")[0])
            words_dict[(str(line)).split("\n")[0]] = -1
    for i in result:
        size = len(i)
        for j in i:
            words_dict[words[j]] = size
    print(words_dict)
    with open(K_MEANS_FILE, 'w') as fp:
        json.dump(words_dict,fp)
    