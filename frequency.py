
import csv
import json
import math
import os


DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
WORDLE_DATA_FILE = os.path.join(DATA_DIR, "freq_map.json")
POSSIBLE_DATA_FILE = os.path.join(DATA_DIR, "possible_words.txt")
POSSIBLE_DATA_FREQ = os.path.join(DATA_DIR, "possible_words_freq.json")
POSSIBLE_DATA_SIGMOID = os.path.join(DATA_DIR, "possible_words_sigmoid.json")

ans_dict={}

def cal():
    fre_dic={}
    possible_list=[]
    #ans_dict={}
    with open(WORDLE_DATA_FILE, 'r') as fp:
        fre_dic=json.loads(fp.read())
        print(len(fre_dic))
    with open(POSSIBLE_DATA_FILE, 'r') as fp:
        for line in fp.readlines():
            possible_list.append(str(line.split('\n')[0]))
    print(possible_list)
    sum = 0
    for i in fre_dic:
        if i in possible_list:
            sum = sum + float(fre_dic[i])
            ans_dict[i] = float(fre_dic[i])
    for i in ans_dict:
        ans_dict[i] = float(ans_dict[i])/sum
    with open(POSSIBLE_DATA_FREQ, 'w') as fp:
        json.dump(ans_dict,fp)

def sigmoid():
    sorted_dict = sorted(ans_dict.items(), key=lambda x: x[1])
    print(sorted_dict)
    sigmoid_dict = {}
    for i in range(len(sorted_dict)):
        x = -10+20*i/(len(sorted_dict)-1)
        sigmoid_dict[sorted_dict[i][0]] = 1/(1 + math.exp(-x))
    print(sigmoid_dict)
    with open(POSSIBLE_DATA_SIGMOID, 'w') as fp:
        json.dump(sigmoid_dict,fp)


if __name__ == "__main__":
    cal()
    sigmoid()