import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from utils.tools import StandardScaler


# 数据处理
class vali_dataProcess(Dataset):
    def __init__(self, df_raw, seq_len, label_len, pred_len, train_times, scale, stock_train_list, mean=None, std=None, timeenc=0, features='MS', freq='h'):
        """
        df_raw: 训练集的原始输入
        scale: 是否需要标准化
        timeenc: 时间编码方式，可选项[0,1]
        features: 预测方式，可选项['MS','M','S']
        freq: 周期，可选项['d','h']
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.train_times = train_times
        self.dataframe_len = seq_len + pred_len + train_times - 1  #dataframe的理想长度
        self.features = features
        self.df = df_raw
        self.freq = freq
        self.timeenc = timeenc
        self.stock_train_list = stock_train_list
        self.scale = scale
        if self.scale == True:
            assert (mean != None).all() and (std != None).all(), "测试集标准化的时候必须要指定训练集的mean和std"
            self.mean_ = mean
            self.std_ = std
        self.__read_data__()


    # 数据预处理函数
    def __read_data__(self):
        df_raw = self.df
        df_raw_1 = pd.DataFrame() #装满足条件的dataframe
        df_len = [] #用来存储每个dataframe的列表，要求每个列表长度都一样为seq_len
        stock_list = []
        stock_test_list = sorted(list(set(df_raw.index.get_level_values(0))))
        
        for stock in stock_test_list:
            # 如果测试集的股票未在训练集中则扔掉
            if stock not in self.stock_train_list:
                continue
            else:
                stock_list.append(stock)
                df_temp = df_raw.loc[stock]
                df_raw_1 = pd.concat([df_raw_1, df_temp], axis=0)
                df_len.append(df_temp.__len__() - self.seq_len - self.pred_len + 1) #股票训练次数
        self.df_raw_1 = df_raw_1
        df_stamp = df_raw_1.index.to_frame().reset_index(drop=True)
        self.df_len = df_len
        self.stock_list = stock_list

        # 任务
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw_1.columns
            df_raw_2 = df_raw_1[cols_data]
        elif self.features == 'S':
            df_raw_2 = df_raw_1[-1:]

        # 标准化
        df_raw_2 = df_raw_2.values.astype('float32')
        self.scaler  = StandardScaler()
        if self.scale:
            self.scaler.fit_(self.mean_,self.std_)
            df_raw_3 = self.scaler.transform(df_raw_2)
        else:
            df_raw_3 = df_raw_2

        # 时间处理
        df_stamp['tradingday'] = pd.to_datetime(df_stamp.tradingday)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.tradingday.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.tradingday.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.tradingday.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.tradingday.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['tradingday'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['tradingday'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 
        
        # 筛选完成的数据变为类自身的属性n
        self.data_x = df_raw_3
        self.data_y = df_raw_3
        self.data_stamp = data_stamp.astype('int32')

    # 自定义索引函数
    def __getitem__(self, index):
        # 每只股票训练次数累加
        cumsum_train_times = [sum(self.df_len[:i+1]) for i in range(len(self.df_len))]

        # 每只股票dataframe的长度和累加
        stockcode_len = [self.seq_len + self.pred_len + i - 1 for i in self.df_len]
        cumsum_stockcode_len = [sum(stockcode_len[:i+1]) for i in range(len(stockcode_len))]

        # 找寻下标
        def Find_index(input_num, sorted_list):
            for i, num in enumerate(sorted_list):
                if input_num < num:
                    return i
            return len(sorted_list)

        # 得到最终索引
        def Get_final_index(index):
            if Find_index(index,cumsum_train_times) == 0:
                a = 0
            else:
                a = cumsum_stockcode_len[Find_index(index,cumsum_train_times)-1]
            return a + index % cumsum_train_times[Find_index(index,cumsum_train_times)-1]

        # 重构索引
        index = Get_final_index(index)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    # 自定义长度函数
    def __len__(self):
        return sum(self.df_len)

    # 逆归一化函数
    def inverse_transform(self, data):
            return self.scaler.inverse_transform(data)
