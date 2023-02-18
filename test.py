import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
FREQ_MAP = os.path.join(DATA_DIR, "possible_words_sigmoid.json")
K_MAP = os.path.join(DATA_DIR, "k_means.json")
PROBLEM_DATA = os.path.join(DATA_DIR, "Problem_C_Data_Wordle.xlsx")


problem_data=pd.read_excel(PROBLEM_DATA,usecols=['1-try','2-tries','3-tries','4-tries','5-tries','6-tries','more'])
#problem_data=pd.read_excel(PROBLEM_DATA,usecols=['4-tries'])
sample_words=pd.read_excel(PROBLEM_DATA,usecols=['Word'])

fre_dic={}
freq_list = []
with open(FREQ_MAP) as fp:
    fre_dic=json.loads(fp.read())
#fre_dic = sorted(fre_dic.items(), key=lambda x: x[0])
for i in sample_words.values:
    freq_list.append(fre_dic[i[0]])

k_dic={}
k_list = []
with open(K_MAP) as fp:
    k_dic=json.loads(fp.read())
for i in sample_words.values:
    k_list.append(k_dic[i[0]])


dic1 = {"freq" : np.array(freq_list),
        "k_mean" : np.array(k_list)}
data_x = pd.DataFrame(dic1)
print(data_x)

x_train, x_test, y_train, y_test = train_test_split(data_x,problem_data, test_size=0.1,random_state=1)
print(x_train)
print(y_train)
"""le = LabelEncoder()
y_train = le.fit_transform(y_train)
print(y_train)"""

def print_precison_recall_f1(y_true, y_pre):
    """打印精准率、召回率和F1值"""
    print("打印精准率、召回率和F1值")
    print(classification_report(y_true, y_pre))
    f1 = round(f1_score(y_true, y_pre, average='macro'), 2)
    p = round(precision_score(y_true, y_pre, average='macro'), 2)
    r = round(recall_score(y_true, y_pre, average='macro'), 2)
    print("Precision: {}, Recall: {}, F1: {} ".format(p, r, f1))

def xgboost_model(x_train,y_train):
    """用XGBoost进行建模，返回训练好的模型"""
    xgboost_clf = XGBClassifier(min_child_weight=6,max_depth=15,
                                objective='multi:softmax',num_class=7)
    print("-" * 60)
    print("xgboost模型：", xgboost_clf)
    xgboost_clf.fit(x_train, y_train)
    # # 打印重要性指数
    #importance_features_top('xgboost', xgboost_clf, x_train)
    # 保存模型
    joblib.dump(xgboost_clf, './model/XGBoost_model_v1.0')
    return xgboost_clf

# 建模
xgboost_clf = xgboost_model(x_train, y_train)
# 预测
pre_y_test = xgboost_clf.predict(x_test)
# 打印测试集的结果信息，包含precision、recall、f1-socre
print("-" * 30, "测试集", "-" * 30)
print(y_test)
print(pre_y_test)
print_precison_recall_f1(y_test, pre_y_test)