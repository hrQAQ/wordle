import json
from logging.handlers import DatagramHandler
import os
import pandas as pd
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib  # 将模型导出所需包

def get_cust_age_stage(birth_year):
    """根据出生年份获取年龄段"""
    age_stage = []
    for i in range(len(birth_year)):
        if int(birth_year[i]) == 0:
            age_stage.append("未知")
        elif int(birth_year[i]) < 1960:
            age_stage.append("60前")
        elif int(birth_year[i]) < 1970:
            age_stage.append("60后")
        elif int(birth_year[i]) < 1980:
            age_stage.append("70后")
        elif int(birth_year[i]) < 1990:
            age_stage.append("80后")
        elif int(birth_year[i]) < 2000:
            age_stage.append("90后")
        elif int(birth_year[i]) >= 2000:
            age_stage.append("00后")
        else:
            age_stage.append("未知")
    return age_stage
def get_top5_onehot(data):
    """对c字段排名top5的进行one hot"""
    # 获取top5的值
    c_top5_counts = data['c'].value_counts()[:5]
    c_top5_names = list(c_top5_counts.keys())
    # 进行one-hot编码，只保留top5的列
    c_one_hot = pd.get_dummies(data['c'])
    c_top5 = c_one_hot[c_top5_names]
    # 将top5的列合并到data中
    data = data.join(c_top5)
    return data

def get_quantile_20_values(input_data):
    """按照分位数切分为20等分"""
    grade = pd.DataFrame(columns=['quantile', 'value'])
    for i in range(0, 21):
        grade.loc[i, 'quantile'] = i / 20.0
        grade.loc[i, 'value'] = input_data.quantile(i / 20.0)
    cut_point = grade['value'].tolist()  # 20等分的分位数的值
    # 对20等分的分位数的值 进行去重
    s_unique = []
    for i in range(len(cut_point)):
        if cut_point[i] not in s_unique:
            s_unique.append(cut_point[i])
    return s_unique

def get_quantile_interregional(s_unique):
    """根据去重后的分位数，构造区间"""
    interregional = []
    for i in range(1, len(s_unique)):
        interregional.append([i, s_unique[i - 1], s_unique[i]])
        if i == len(s_unique) - 1 and len(interregional) < 20:
            interregional.append([i + 1, s_unique[i], s_unique[i]])
    return interregional

def get_current_level(item_data,interregional):
    """根据分位数区间获取当前数所对应的的级别"""
    level = 0
    for i in range(len(interregional)):
        if item_data >= interregional[i][1] and item_data <interregional[i][2]:
            level = interregional[i][0]
            break
        elif interregional[i][1] == interregional[i][2]:
            level = interregional[i][0]
            break
    return level

def get_division_level(input_data):
    """根据分位数划分对应级别"""
    # 获取去重后20等分的分位数的值
    s_unique = get_quantile_20_values(input_data)
    # 构造分位数区间，输出格式[index,下限，上限]  区间为左闭右开
    interregional = get_quantile_interregional(s_unique)
    # 根据分位数区间对数据划分不同等级
    quantile_20_level = []
    for item in input_data:
        quantile_20_level.append(get_current_level(item, interregional))
    return quantile_20_level

def pre_processing(data):
    """对数据进行预处理"""
    # 1. 增加衍生变量
    # 年龄
    data['年龄'] = get_cust_age_stage(data['出生年份'])
    # 本月平均时长
    data['本月平均时长'] = data['本月时长'].div(data['本月次数'],axis=0)
    data['g'] = data['a'] - data['b']

    # 2. 填充数据
    col_name_0 = ['a', 'b','g', 'k']  # 需要填充为数字0的指标名
    values = {}
    for i in col_name_0:
        values[i] = 0
    # 不加inplace=True，数据不会被填充
    data.fillna(value=values, inplace=True)
    data.fillna({'m':'未知', 'z':'未知'}, inplace=True)  # m/z列需要填充为字符串
    # 对c指标进行one-hot处理
    data = get_top5_onehot(data)
    # 3. 分级化
    col_name_level = ['d', 'e', 'f']
    for i in range(len(col_name_level)):
        new_col_name = col_name_level[i] + "_TILE20"
        data[new_col_name] = get_division_level(data[col_name_level[i]])
    return DatagramHandler

def get_model_columns(input_data):
    """获取建模指标列名，列表类型"""
    total_col_names = input_data.columns
    del_col_names = ['a','b','c']
    model_col_names = [i for i in total_col_names if i not in del_col_names]
    return model_col_names

def importance_features_top(model_str, model, x_train):
    """打印模型的重要指标，排名top10指标"""
    print("打印XGBoost重要指标")
    feature_importances_ = model.feature_importances_
    feature_names = x_train.columns
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)], 
                                  columns=['a', 'b'])
    importance_col_desc = importance_col.sort_values(by='b', ascending=False)
    print(importance_col_desc.iloc[:10, :])

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
                                objective='multi:softmax',num_class=5)
    print("-" * 60)
    print("xgboost模型：", xgboost_clf)
    xgboost_clf.fit(x_train, y_train)
    # # 打印重要性指数
    importance_features_top('xgboost', xgboost_clf, x_train)
    # 保存模型
    joblib.dump(xgboost_clf, './model/XGBoost_model_v1.0')
    return xgboost_clf


DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
FREQ_MAP = os.path.join(DATA_DIR, "freq_map.json")


fre_dic={}
with open(FREQ_MAP) as fp:
    fre_dic=json.loads(fp.read())
df = pd.DataFrame(fre_dic)


data = pd.read_excel(filename)
# 数据预处理，包括填充数据，增加衍生变量、分级化、top打横
data_processed = pre_processing(data)
# 根据业务删除某些变量,获取建模所需指标
model_col_names = get_model_columns(input_data)
model_data = data_processed[model_col_names]
# 将数据拆分为输入数据和输出数据
data_y = model_data['label']
data_x = model_data.drop(['label'], axis=1)
# 数据集拆分为训练集和测试集两部分  使用随机数种子，确保可以复现
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,
                                                    test_size=0.3,random_state=1)
# 建模
xgboost_clf = xgboost_model(x_train, y_train)
# 预测
pre_y_test = xgboost_clf.predict(x_test)
# 打印测试集的结果信息，包含precision、recall、f1-socre
print("-" * 30, "测试集", "-" * 30)
print_precison_recall_f1(y_test, pre_y_test)