from utils.sqlite_func import *
import pandas
import os
os.chdir(r'C:\Users\user\Desktop\股票测试代码')

# 文件根目录
path = r'd:\python_informer\pred_csv\long_term_forecast_Informer\交通运输'    #需要修改
# 模型
model = 'Informer'    #需要修改
# 行业
industry = '交通运输'   #需要修改
stock = None
Ashares = False
if Ashares:
    industry_name = 'ashares'
else:
    industry_df = pd.read_csv('./industry_ID_DataFrame.csv')
    industry_df.set_index('Industry', inplace=True)
    industry_name = industry_df.loc[industry][0]



# # 读取到数据库
# if __name__ == '__main__':
#     file_list = [os.path.join(k,m) for k in [os.path.join(path,i) for i in os.listdir(path)] for m in os.listdir(k)]
#     for file in file_list:
#         df = pd.read_csv(file)
#         df_to_db(df,table_name=f'{model}_{industry_name}')


# 生成回测数据
if __name__ == '__main__':
    start_ = '20150801'
    end_ = '20230801'
    # # 生成选股表
    df = read_df_from_db(start_day=start_,end_day=end_,table_name=f'{model}_{industry_name}')

    # # 自定义分组
    # df_ = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[:int(len(x)*0.5),:])
    # df_ = df_[['tradingday','code']]
    # df_['W'] = 1
    # df_.to_csv('第一组.csv',index=False)


    # 分成五组
    df1 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[:int(len(x)*0.2),:])
    df1 = df1[['tradingday','code']]
    df1['W'] = 1
    df1.to_csv('第一组.csv',index=False)

    df2 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[int(len(x)*0.2):int(len(x)*0.4),:])
    df2 = df2[['tradingday','code']]
    df2['W'] = 1
    df2.to_csv('第二组.csv',index=False)

    df3 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[int(len(x)*0.4):int(len(x)*0.6):,:])
    df3 = df3[['tradingday','code']]
    df3['W'] = 1
    df3.to_csv('第三组.csv',index=False)

    df4 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[int(len(x)*0.6):int(len(x)*0.8),:])
    df4 = df4[['tradingday','code']]
    df4['W'] = 1
    df4.to_csv('第四组.csv',index=False)

    df5 = df.groupby(['tradingday'],group_keys=False).apply(lambda x:x.iloc[int(len(x)*0.8):,:])
    df5 = df5[['tradingday','code']]
    df5['W'] = 1
    df5.to_csv('第五组.csv',index=False)