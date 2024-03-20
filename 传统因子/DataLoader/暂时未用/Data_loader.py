import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from utils.tools import StandardScaler


# 数据处理
class DataProcess(Dataset):
    def __init__(self, df_raw, seq_len, label_len, pred_len, train_len, scale, flag, df_fit=None, timeenc=0, features = 'MS', freq='h'):
        """
        seq_len: seq长度
        label_len: label长度
        pred_len: 预测长度
        train_len: 训练长度
        scale: 是否进行缩放
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.train_len = train_len
        assert flag in ['train','test']
        self.flag = flag
        self.df_fit = df_fit

        self.dataframe_len = seq_len + pred_len + train_len - 1  #回测天数
        self.features = features
        self.df = df_raw
        self.freq = freq
        self.timeenc = timeenc
        self.scale = scale
        self.__read_data__()

    # 数据预处理函数
    def __read_data__(self):
        df_len_ = []  
        stock_list_ = []

        df_raw_1 = pd.DataFrame() #用来存放每个股票的DataFrame
        df_stamp = pd.DataFrame()

        # 股票代码库，用来存放行业类的所有股票代码
        stock_list = sorted(list(set(self.df.index.get_level_values(0).tolist())))

        # 对df_raw重新做筛选
        for stock in stock_list:
            # 筛选出满足要求的stock的训练样本不能太短
            if self.df.loc[stock].__len__() < self.dataframe_len*0.8:
                continue
            else:
                # 筛选出feature和label
                df_temp_1 = self.df.loc[stock]
                stamp = self.df.loc[stock].index.to_frame().reset_index(drop=True)

                # 用列表来存储每只股票的feature和label长度
                if self.dataframe_len != df_temp_1.shape[0]:
                     continue
                else:
                    df_len_.append(df_temp_1.shape[0])
                    df_raw_1 = pd.concat([df_raw_1, df_temp_1], axis=0)
                    df_stamp = pd.concat([df_stamp, stamp], axis=0)
                    stock_list_.append(stock) #用来存放筛选完成后的股票列表

        # 所有股票的dataframe长度都一样长 ！
        assert (len(set(df_len_)) == 1) and (stock_list_.__len__() == df_len_.__len__())
        self.stock_list_ = stock_list_
        self.stock_len = len(stock_list_) #筛选完成后股票的数量

        # 根据做筛选，经过筛选后已经
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw_1.columns  #取出所有列的列名
            df_raw_2 = df_raw_1[cols_data]
        elif self.features == 'S':
            df_raw_2 = df_raw_1[-1:]

        # 根据训练集和测试集进行不同的标准化
        if self.flag == 'train':
            df_raw_2 = df_raw_2.values.astype('float32')
            self.df_fit_out = df_raw_2
            self.scaler  = StandardScaler()
            if self.scale:
                self.scaler.fit(df_raw_2)
                df_raw_3 = self.scaler.transform(df_raw_2)
            else:
                df_raw_3 = df_raw_2
        else:
            df_raw_2 = df_raw_2.values.astype('float32')
            self.scaler  = StandardScaler()
            if self.scale:
                self.scaler.fit(self.df_fit)
                df_raw_3 = self.scaler.transform(df_raw_2)
            else:
                df_raw_3 = df_raw_2


        # 时间获取
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
        
        # 筛选完成的数据变为类自身的属性
        self.data_x = df_raw_3
        self.data_y = df_raw_3
        self.data_stamp = data_stamp.astype('int32')

    # 自定义索引函数
    def __getitem__(self, index):
        index = (index // self.train_len)*self.dataframe_len + index%self.train_len
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
        return self.train_len*self.stock_len

    # 逆归一化函数
    def inverse_transform(self, data):
            return self.scaler.inverse_transform(data)
    

