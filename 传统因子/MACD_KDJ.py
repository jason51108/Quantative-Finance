# 数据读取
from sqlalchemy import create_engine
import numpy as np
from utils.sqlite_func import *
from DataLoader.GetData import *
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir(r'C:\Users\user\Desktop\股票测试代码\传统因子')

# 回测
def AlphaCheak(root_path):
    for file_directory in [os.path.join(root_path,i) for i in os.listdir(root_path) if i.endswith(".csv")]:
        with subprocess.Popen(r'C:\Users\user\Desktop\SqlDbx\回测工具\AlphaCheck.exe', stdin=subprocess.PIPE, text=True) as proc:
            # 通过标准输入传递参数，每个参数后面加上换行符
            proc.stdin.write(f'{file_directory}\n')
            proc.stdin.write(f'{100000000}\n')
            proc.stdin.write('SH000852\n')
            proc.stdin.flush() # 确保所有数据都被写入
            # 等待进程完成
            proc.wait()

# MACD策略
def MACD(start_,end_,code):
    df = Get_index(engine, start_, end_,code).reset_index() #向前多少天，做训练和预测
    data = np.array(df.close) 
    ndata = len(data)
    m, n, T = 12, 26, 9
    EMA1 = np.copy(data)
    EMA2 = np.copy(data)
    f1 = (m-1)/(m+1)
    f2 = (n-1)/(n+1)
    f3 = (T-1)/(T+1)
    for i in range(1, ndata):
        EMA1[i] = EMA1[i-1]*f1 + EMA1[i]*(1-f1)
        EMA2[i] = EMA2[i-1]*f2 + EMA2[i]*(1-f2)
    df['ma1'] = EMA1
    df['ma2'] = EMA2
    DIF = EMA1 - EMA2
    df['DIF'] = DIF
    DEA = np.copy(DIF)
    for i in range(1, ndata):
        DEA[i] = DEA[i-1]*f3 + DEA[i]*(1-f3)
    df['DEA'] = DEA

    # 判断金叉死叉
    df['macd_金叉死叉'] = ''
    macd_position = df['DIF'] > df['DEA']
    df.loc[macd_position[(macd_position == True) & (macd_position.shift() == False)].index, 'macd_金叉死叉'] = '金叉'
    df.loc[macd_position[(macd_position == False) & (macd_position.shift() == True )].index, 'macd_金叉死叉'] = '死叉'

    # 设置查询条件
    query_buy_1 = df['macd_金叉死叉'] == '金叉'
    query_sell_1 = df['macd_金叉死叉'] == '死叉'

    df['Position'] = None
    df.loc[query_buy_1, 'Position'] = 1
    df.loc[query_sell_1, 'Position'] = 0
    df['Position'].fillna(method='ffill', inplace=True)
    df.dropna(subset=['Position'],inplace=True)

    whether_path_exist('./结果/MACD')
    # 预测组
    df_pred = df[df['Position'] == 1][['tradingday','code']]
    df_pred['w'] = 1
    df_pred.to_csv('./结果/MACD/pred.csv',index=False)

# KDJ策略
def KDJ(start_,end_,code):
    df = Get_index(engine, start_, end_,code).reset_index() #向前多少天，做训练和预测
    low_list=df['low'].rolling(window=9).min()
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = df['high'].rolling(window=9).max()
    high_list.fillna(value=df['high'].expanding().max(), inplace=True)

    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['KDJ_K'] = rsv.ewm(com=2).mean()
    df['KDJ_D'] = df['KDJ_K'].ewm(com=2).mean()
    df['KDJ_J'] = 3 * df['KDJ_K'] - 2 * df['KDJ_D']

    # 判断金叉死叉
    df['KDJ_金叉死叉'] = ''
    kdj_position = df['KDJ_K'] > df['KDJ_D']
    df.loc[kdj_position[(kdj_position == True) & (kdj_position.shift() == False)].index, 'KDJ_金叉死叉'] = '金叉'
    df.loc[kdj_position[(kdj_position == False) & (kdj_position.shift() == True)].index, 'KDJ_金叉死叉'] = '死叉'

    # 设置查询条件
    query_buy_2 = df['KDJ_金叉死叉'] == '金叉'
    query_sell_2 = df['KDJ_金叉死叉'] == '死叉'

    df['Position'] = None
    df.loc[query_buy_2, 'Position'] = 1
    df.loc[query_sell_2, 'Position'] = 0
    df['Position'].fillna(method='ffill', inplace=True)
    df.dropna(subset=['Position'],inplace=True)

    whether_path_exist('./结果/KJD')
    # 预测组
    df_pred = df[df['Position'] == 1][['tradingday','code']]
    df_pred['w'] = 1
    df_pred.to_csv('./结果/KJD/pred.csv',index=False)


if __name__ == '__main__':
    # 数据库设置
    engine = create_engine("mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server")
    start_ = '20141017'
    end_ = '20230801'
    code = 'SZ399906'

    # MACD
    MACD(start_,end_,code)
    KDJ(start_,end_,code)