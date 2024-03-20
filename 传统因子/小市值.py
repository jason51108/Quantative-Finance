# 数据读取
from sqlalchemy import create_engine
import numpy as np
from utils.sqlite_func import *
from DataLoader.GetData import *
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

# 检查是否存在文件
def whether_path_exist(folder_path): # 如果文件路径不存在则新建路径
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'已新建路径:  {folder_path}')
    else:
        # print(f'路径已存在:  {folder_path}')
        pass

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

# 小市值做筛选
def pick_stocks_fixed(m, n, start_, end_):
    engine = create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')
    quary = f'''WITH RankedStocks AS (
                SELECT
                    a.tradingday,
                    a.code,
                    b.totalMV,
                    b.industry,
                    ROW_NUMBER() OVER (PARTITION BY a.tradingday ORDER BY b.totalMV) AS Rank
                FROM
                    daily..daybar a
                JOIN
                    daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                WHERE
                    datediff(day, listeddate, a.tradingday) > 180 AND
                    tradeable = 1 AND
                    volume > 0 AND
                    a.tradingday BETWEEN '{start_}' AND '{end_}'
            )
            SELECT
                tradingday,
                code,
                totalMV,
                industry
            FROM
                RankedStocks
            WHERE
                Rank <= {n};'''
    df = pd.read_sql(quary, engine)
    # 将 'tradingday' 列转换为日期类型
    df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
    df = df.sort_values(by=['code', 'tradingday'])
    result_df = df.groupby('code').apply(lambda x: x.iloc[::m]).reset_index(drop=True)
    result_df = result_df[result_df.columns[:2]]
    result_df['W'] = 1
    result_df.sort_values('tradingday',inplace=True)
    return result_df

# 小市值做筛选(按照比例做筛选)
def pick_stocks_ratio(m, ratio, start_, end_):
    engine = create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')
    quary = f"""
            WITH RankedStocks AS (
            SELECT
                a.tradingday,
                a.code,
                b.totalMV,
                b.industry,
                PERCENT_RANK() OVER (PARTITION BY a.tradingday ORDER BY b.totalMV) AS PercentileRank
            FROM
                daily..daybar a
            JOIN
                daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
            WHERE
                datediff(day, listeddate, a.tradingday) > 180 AND
                tradeable = 1 AND
                volume > 0 AND
                a.tradingday BETWEEN '{start_}' AND '{end_}' AND
                b.totalMV < 5000000000
        )
        SELECT
            tradingday,
            code,
            totalMV,
            industry
        FROM
            RankedStocks
        WHERE
            PercentileRank <= {ratio};
            """
    df = pd.read_sql(quary, engine)
    # 将 'tradingday' 列转换为日期类型
    df['tradingday'] = pd.to_datetime(df['tradingday'], format='%Y%m%d')
    df = df.sort_values(by=['code', 'tradingday'])
    result_df = df.groupby('code').apply(lambda x: x.iloc[::m]).reset_index(drop=True)
    result_df = result_df[result_df.columns[:2]]
    result_df['W'] = 1
    result_df.sort_values('tradingday',inplace=True)
    return result_df



if __name__ == '__main__':
    os.chdir(r'C:\Users\user\Desktop\股票测试代码\传统因子')
    whether_path_exist(r'./结果/小市值')
    m = 5 # 持仓时间
    n = 20 # 小市值前多少只股票
    ratio = 0.05
    start_ = '20141017'
    end_ = '20230801'

    df = pick_stocks_fixed(m, n, start_, end_)
    # df = pick_stocks_ratio(m, ratio, start_, end_)
    df.to_csv(r'./结果/小市值/小市值.csv',index = False)
    AlphaCheak(r'./结果/小市值/')