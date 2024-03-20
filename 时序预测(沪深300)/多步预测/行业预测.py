# torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from DataLoader.Train_loader import train_dataProcess
from DataLoader.Vali_loader import vali_dataProcess
from DataLoader.Test_muti_loader import test_dataProcess
from utils.tools import EarlyStopping, adjust_learning_rate

# 模型导入
from models import Transformer, Informer, TimesNet, Pyraformer, Nonstationary_Transformer, DLinear, FEDformer, LightTS, Reformer, ETSformer
from utils.tools import whether_path_exist
from utils.sqlite_func import *

# 数据读取
from sqlalchemy import create_engine
from utils.sqlite_func import *
from DataLoader.GetData import *

# 数据分析
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# config参数
import argparse
import subprocess

# 时间进度条
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import warnings
warnings.filterwarnings("ignore")
os.chdir(r'C:\Users\user\Desktop\股票测试代码\时序预测(沪深300)\多步预测')

# 模型参数传递函数
def return_par(model_name, epochs, batch_size, seq_len, label_len, pred_len, train_len, n, m, features, scale, d_model, lr, enc_in, dec_in, c_out):
    parser = argparse.ArgumentParser(description='股票预测')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--features', type=str, default=features,help='预测任务, options:[M, S, MS]; M:多对多, S:单对单, MS:多对单')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--model', type=str, default=model_name, help='模型名字, options: [Autoformer, Transformer, TimesNet, Informer, Pyraformer .....]')
    parser.add_argument('--seq_len', type=int, default = seq_len, help='input sequence length')
    parser.add_argument('--label_len', type=int, default = label_len, help='start token length')
    parser.add_argument('--pred_len', type=int, default = pred_len, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default = enc_in, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default = dec_in, help='decoder input size')
    parser.add_argument('--c_out', type=int, default = c_out , help='output size')
    parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', default=False, action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')
    parser.add_argument('--learning_rate', type=str, default=lr, help='learning rate')
    parser.add_argument('--patience', type=int, default=8, help='learning rate')
    

    # Informer
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)        
    # Times_net
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=2, help='for Inception')
    args = parser.parse_args(args=[])
    return args

# 模型生成函数
def create_model(args):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {
            'TimesNet': TimesNet,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'Pyraformer': Pyraformer,
        }
    return model_dict[args.model].Model(args).to(device)

# 验证函数
def vali(vali_loader):
    model.eval() #注意model必须为全局变量
    with torch.no_grad():
            loss_ = 0
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device) #device也为全局变量，因此无需在函数内部传递
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, label_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

                # 前向传播
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #注意,在原始模型里,已经对outputs在L维度上做了切分

                # 根据特征来切割
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, label_len:, f_dim:].to(device)

                # 计算损失
                loss = loss_fc(outputs, batch_y)
                loss_ += loss.item()
            epoch_loss = loss_ / len(vali_loader)
    return epoch_loss

# 训练和测试函数
def train_test(industry, day, args):
    ##=====数据生成
    # 数据集
    df_train = Get_data(day, n, data_type='Timeseries', industry = industry, stock = stock, Ashares = Ashares)
    train_data = train_dataProcess(df_train, seq_len, label_len, pred_len, train_len, scale=scale)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, drop_last=False)
    # 验证集
    df_vali = Get_data(day, int(m), data_type='Timeseries', industry = industry, stock = stock, Ashares = Ashares)
    vali_data = vali_dataProcess(df_vali, seq_len, label_len, pred_len, train_len*0.1, scale, train_data.stock_list) if scale == False else vali_dataProcess(df_vali, seq_len, label_len, pred_len, train_len*0.1, scale, train_data.stock_list, mean=train_data.mean,std=train_data.std)
    vali_loader = DataLoader(vali_data, batch_size = int(train_len*0.1), shuffle=True, drop_last=False)
    # 预测集
    next_day = (pd.to_datetime(day) + relativedelta(months=1)).strftime('%Y%m%d') #下一次训练的时间
    time_feature = Get_All_Tradingday(date=day,n = pred_len, forward=False)[:22]
    df_test = Get_data(day, seq_len+10, data_type='Timeseries', industry = industry, stock = stock, Ashares = Ashares)  #预测时间往前推seq的时间
    test_data = test_dataProcess(df_test, seq_len, label_len, pred_len, scale, train_data.stock_list, time_feature) if scale == False else test_dataProcess(df_test, seq_len, label_len, pred_len, scale, train_data.stock_list, time_feature, mean=train_data.mean,std=train_data.std)
    test_loader = DataLoader(test_data, batch_size = 1, shuffle=False, drop_last=False) #注意batch_size也要进行更改

    # 早停函数
    early_stopping = EarlyStopping(args.patience, verbose=True)

    # 是否需要训练
    if need_train:
        #模型调到训练模式
        model.train() 
        pbar_train = tqdm(range(epochs))
        for epoch in pbar_train:
            loss_ = 0
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, label_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

                # 前向传播
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #注意,在原始模型里,已经对outputs在L维度上做了切分

                # 根据特征来切割
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, label_len:, f_dim:].to(device)

                # 计算损失
                loss = loss_fc(outputs, batch_y)
                loss.backward()
                optimizer.step()
                loss_ += loss.item()
            epoch_loss = loss_ / len(train_loader)

            vali_loss = vali(vali_loader)
            train_s = "- train ==> time:{} - epoch:{} - train_loss:{} - vali_loss:{}".format(day, epoch+1, epoch_loss, vali_loss)
            pbar_train.set_description(train_s)

            early_stopping(vali_loss, model, os.path.join(save_path,f'{day}.pth'))
            if early_stopping.early_stop:
                break #打破所有循环
            adjust_learning_rate(optimizer, epoch + 1, args)

        #生成模型文件
        model_path = os.path.join(save_path,f'{day}.pth') 
        torch.save(model.state_dict(), model_path)


        # 调整为测试模式
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, label_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

                # 预测
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 根据特征来切割
                outputs = outputs.detach().cpu().numpy().squeeze(0)
                batch_y = batch_y.detach().cpu().numpy().squeeze(0)

                # 反归一化
                if test_data.scale and args.inverse:
                    # 通通返回二维数据，第一个batch总是为1，因为我batchsize设置为1
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)
                
                # 最后按照维度做切分
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, f_dim:]
                batch_y = batch_y[label_len:, f_dim:]

                # 输出结果
                df = pd.DataFrame({'tradingday':time_feature,'pred':outputs.flatten(),'code':test_data.stock_list[i]})
                df_out = df[df['tradingday'] < next_day]
                # df_out.to_csv(os.path.join(os.path.join(csv_path,f'{day}'),f'df_trade-{day}-{test_data.stock_list[i]}.csv'),index=False)
                df_to_db(df_out,table_name=table_name) #策略-模型-行业
        torch.cuda.empty_cache() 

    else:
        print(day)
        # 模型参数地址
        model_path = os.path.join(save_path,f'{day}.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()   
        with torch.no_grad():
            # 每次需要进行清楚
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                # Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, label_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

                # 预测
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 根据特征来切割
                outputs = outputs.detach().cpu().numpy().squeeze(0)
                batch_y = batch_y.detach().cpu().numpy().squeeze(0)

                # 反归一化
                if test_data.scale and args.inverse:
                    # 通通返回二维数据，第一个batch总是为1，因为我batchsize设置为1
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)
                
                # 最后按照维度做切分
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, f_dim:]
                batch_y = batch_y[label_len:, f_dim:]

                # 输出结果
                df = pd.DataFrame({'tradingday':time_feature,'pred':outputs.flatten(),'code':test_data.stock_list[i]})
                df_out = df[df['tradingday'] < next_day]
                # df_out.to_csv(os.path.join(os.path.join(csv_path,f'{day}'),f'df_trade-{day}-{test_data.stock_list[i]}.csv'),index=False)
                df_to_db(df_out,table_name=table_name) #策略-模型-行业
        torch.cuda.empty_cache() 

# 进行分组排序
def order_1(path, table_name):
    # path = f'../回测结果/{name_}/{args.task_name}_{args.model}_scale_{scale}/{industry}/' if industry != None else f'./pred_csv/{name_}/{args.task_name}_{args.model}_scale_{scale}/Ashares/'
    whether_path_exist(path)

    # 生成选股表
    df = read_df_from_db(start_day=pd.to_datetime(start_).strftime('%Y-%m-%d'),end_day=pd.to_datetime(end_).strftime('%Y-%m-%d'),table_name = table_name)

    # 分成五组
    df1 = df.groupby('tradingday').apply(lambda x: x[x['pred'] > x['pred'].quantile(0.8)]).reset_index(drop=True)
    df1 = df1[['tradingday','code']]
    df1['W'] = 1
    df1.to_csv(path+'第一组.csv',index=False)

    df2 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.6)) & (x['pred'] <= x['pred'].quantile(0.8))]).reset_index(drop=True)
    df2 = df2[['tradingday','code']]
    df2['W'] = 1
    df2.to_csv(path+'第二组.csv',index=False)

    df3 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.4)) & (x['pred'] <= x['pred'].quantile(0.6))]).reset_index(drop=True)
    df3 = df3[['tradingday','code']]
    df3['W'] = 1
    df3.to_csv(path+'第三组.csv',index=False)

    df4 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.2)) & (x['pred'] <= x['pred'].quantile(0.4))]).reset_index(drop=True)
    df4 = df4[['tradingday','code']]
    df4['W'] = 1
    df4.to_csv(path+'第四组.csv',index=False)

    df5 = df.groupby('tradingday').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.2)]).reset_index(drop=True)
    df5 = df5[['tradingday','code']]
    df5['W'] = 1
    df5.to_csv(path+'第五组.csv',index=False)

    df6 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[:,:])
    df6 = df6[['tradingday','code']]
    df6['W'] = 1
    df6.to_csv(path+'第六组.csv',index=False)

# 进行分组排序，累成后返回结果
def order_2(path, table_name):
    whether_path_exist(path)
    df = read_df_from_db(start_day=pd.to_datetime(start_).strftime('%Y-%m-%d'),end_day=pd.to_datetime(end_).strftime('%Y-%m-%d'),table_name=table_name)

    # 对数据做处理
    df['tradingday'] = pd.to_datetime(df['tradingday'])
    df['cumulative_pred'] = df.groupby([df['tradingday'].dt.to_period("M"), 'code'])['pred'].transform(lambda x: (1 + x).cumprod())
    last_day_data = df.groupby([df['tradingday'].dt.to_period("M"), 'code']).apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    df = pd.merge(df, last_day_data[['tradingday', 'code', 'cumulative_pred']], on=['tradingday', 'code'], suffixes=('', '_last_day'), how='left')
    df['cumulative_pred_last_day'] = df.groupby([df['tradingday'].dt.to_period("M"), 'code'])['cumulative_pred_last_day'].fillna(method='bfill')
    df.drop(columns=['pred', 'cumulative_pred'], inplace=True)
    df.rename(columns={'cumulative_pred_last_day':'pred'},inplace=True)
    
    # 分成五组
    df1 = df.groupby('tradingday').apply(lambda x: x[x['pred'] > x['pred'].quantile(0.8)]).reset_index(drop=True)
    df1 = df1[['tradingday','code']]
    df1['W'] = 1
    df1.to_csv(path+'第一组.csv',index=False)

    df2 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.6)) & (x['pred'] <= x['pred'].quantile(0.8))]).reset_index(drop=True)
    df2 = df2[['tradingday','code']]
    df2['W'] = 1
    df2.to_csv(path+'第二组.csv',index=False)

    df3 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.4)) & (x['pred'] <= x['pred'].quantile(0.6))]).reset_index(drop=True)
    df3 = df3[['tradingday','code']]
    df3['W'] = 1
    df3.to_csv(path+'第三组.csv',index=False)

    df4 = df.groupby('tradingday').apply(lambda x: x[(x['pred'] > x['pred'].quantile(0.2)) & (x['pred'] <= x['pred'].quantile(0.4))]).reset_index(drop=True)
    df4 = df4[['tradingday','code']]
    df4['W'] = 1
    df4.to_csv(path+'第四组.csv',index=False)

    df5 = df.groupby('tradingday').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.2)]).reset_index(drop=True)
    df5 = df5[['tradingday','code']]
    df5['W'] = 1
    df5.to_csv(path+'第五组.csv',index=False)

    df6 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[:,:])
    df6 = df6[['tradingday','code']]
    df6['W'] = 1
    df6.to_csv(path+'第六组.csv',index=False)

#使用 subprocess.Popen 运行 exe 文件并传递参数
def AlphaCheak(root_path):
    # 判定需要时csv文件才行
    for file_directory in [os.path.join(root_path,i) for i in os.listdir(root_path) if i.endswith(".csv")]:
        with subprocess.Popen(r'C:\Users\user\Desktop\SqlDbx\回测工具\AlphaCheck.exe', stdin=subprocess.PIPE, text=True) as proc:
            # 通过标准输入传递参数，每个参数后面加上换行符
            proc.stdin.write(f'{file_directory}\n')
            proc.stdin.write(f'{100000000}\n')
            proc.stdin.write('SH000852\n')
            proc.stdin.flush() # 确保所有数据都被写入
            # 等待进程完成
            proc.wait()

# 聚合查看结果
def aggregate(root_path):
    dict_ = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    result_df = pd.DataFrame()
    for file_directory in [os.path.join(root_path,i) for i in os.listdir(root_path) if i.endswith(".xls")]:
        df = pd.read_excel(file_directory,sheet_name='PnL')
        result_df['TradingDay'] = df['TradingDay']
        result_df[f'alpha-{dict_[file_directory[-13]]}'] = df['Alpha']
    result_df.to_csv(os.path.join(root_path,'plot_result.csv'),index=False)

# 程序运行
if __name__ == '__main__':
    ##===== 回测函数(可调)
    name_ = 'Mult_Strategy_1' #策略名称
    need_train = True #是否需要训练
    max_threads = 4 #最大线程数

    start_ = '20210501'
    end_ = '20231201'
    selecte_train_day = 1 #每月几号进行训练
    industry = '交通运输'
    stock = None #'SH600005'或None
    Ashares = False #True或False

    # 生成industry_name
    if Ashares:
        industry_name = 'Ashares'
    else:
        industry_df = pd.read_csv('./industry_ID_DataFrame.csv')
        industry_df.set_index('Industry', inplace=True)
        industry_name = industry_df.loc[industry][0]

    ##===== 模型参数(可调)
    model_name = 'Transformer'
    epochs = 50
    batch_size = 64
    seq_len = 22
    label_len = 11
    pred_len = 22

    train_len = 40
    n = seq_len  +  pred_len + train_len - 1  #回测天数
    m = seq_len  +  pred_len + train_len*0.1 - 1  #验证集天数
    features = 'MS'

    scale = True
    d_model = 128
    lr = 1e-4 

    enc_in = 6    # encoder输入特征数
    dec_in = 6    # decoder输入特征数
    c_out = 6     # 是在输出结果后面进行切割

    ##===== 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = return_par(model_name, epochs, batch_size, seq_len, label_len, pred_len, train_len, n, m, features, scale, d_model, lr, enc_in, dec_in, c_out)
    model = create_model(args)
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ##===== 数据库和cheakpoint地址
    table_name = f'{name_}_mn_{model_name}_in_{industry_name}_st_{start_}_end_{end_}_fs_{features}_ep_{epochs}_bz_{batch_size}_sq_{seq_len}_lb_{label_len}_pd_{pred_len}_tl_{train_len}_sl_{scale}_dm_{d_model}_k_{args.top_k}_nk_{args.num_kernels}' 
    save_path = f'./checkpoints/'+ table_name #模型参数地址
    whether_path_exist(save_path) 

    ##===== 训练和预测
    # 如果数据库存在文件数据则不进行训练
    if whether_exist_table(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = table_name):
        print(f'{industry}模型已训练好，且预测数据存在数据集，直接进行分组排序!')
        order_1(path = f'../回测结果/每日换仓/{table_name}/', table_name=table_name)
        order_2(path = f'../回测结果/每月换仓/{table_name}/', table_name=table_name)
        AlphaCheak(f'../回测结果/每日换仓/{table_name}/')
        AlphaCheak(f'../回测结果/每月换仓/{table_name}/')
        aggregate(f'../回测结果/每日换仓/{table_name}/')
        aggregate(f'../回测结果/每月换仓/{table_name}/')
        print('分组完成，已存在回测结果文件夹下')
    else:
        start = time.time()
        # drop_tabel(table_name=table_name) #最好手动删除表
        print('*'*30+f'正在训练{industry}行业')
        day_list = [str(year) + '{:02d}'.format(month) + '{:02d}'.format(selecte_train_day) for year in range(int(start_[:4]), int(end_[:4])+1) for month in range(1, 13)]
        day_list = list(filter(lambda x: eval(x) >= eval(start_) and eval(x) < eval(end_), day_list))
        
        # 单线程
        for day in day_list:
            try:
                train_test(industry, day, args)
            except:
                print(f'{industry}行业在{day}之前发生了变化导致后续预测数据为空')
                continue
        ## 多线程运行
        # with ThreadPoolExecutor(max_threads) as executor:
        #     futures = [executor.submit(train_test, day) for day in day_list]
        end = time.time()
        print(f'训练{table_name}所花费的时间为{(end-start)/3600}小时')
        order_1(path = f'../回测结果/每日换仓/{table_name}/', table_name=table_name)
        order_2(path = f'../回测结果/每月换仓/{table_name}/', table_name=table_name)
        AlphaCheak(f'../回测结果/每日换仓/{table_name}/')
        AlphaCheak(f'../回测结果/每月换仓/{table_name}/')
        aggregate(f'../回测结果/每日换仓/{table_name}/')
        aggregate(f'../回测结果/每月换仓/{table_name}/')
    