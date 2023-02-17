import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import statsmodels.graphics.tsaplots as tsaplots
import seaborn as sns

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
IMG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "img",
)

WORDLE_DATA_FILE = os.path.join(DATA_DIR, "Problem_C_Data_Wordle.xlsx")

def get_wordle_data():
    return pd.read_excel(WORDLE_DATA_FILE)

def arima():
    data = get_wordle_data()
    date = data["Date"]
    total_num_repo = data["Total-num-repo"]
    trend = [date, total_num_repo]
    # 绘制趋势图
    plt.plot(date, total_num_repo)
    plt.savefig(os.path.join(IMG_DIR, "arima.png"))
    # 绘制自相关图
    tsaplots.plot_acf(total_num_repo, lags=20)   # lags为滞后阶数
    plt.savefig(os.path.join(IMG_DIR, "arima_acf.png"))
    # 绘制偏自相关图
    tsaplots.plot_pacf(total_num_repo, lags=20)
    plt.savefig(os.path.join(IMG_DIR, "arima_pacf.png"))
    #平稳性检测
    from statsmodels.tsa.stattools import adfuller as ADF
    print(u'原始序列的ADF检验结果为:', ADF(trend[u'Total-num-repo']))    

if __name__ == "__main__":
    arima()