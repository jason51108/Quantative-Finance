import sqlite3
import pandas as pd
from utils.tools import whether_path_exist
import os


# 注意注意注意！
# 当进行数据库连接时，conn = sqlite3.connect('test.db')，如果数据库不存在，那么它就会被创建
def create_db(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast'):
    '''
    功能: 新建一个数据库
    输入: 
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
    Example:
    >>> create_db(db_file_path="../daily_stock_daybar/", db_name = 'long_term_forecast')
    '''
    conn = sqlite3.connect(os.path.join(db_file_path, f'{db_name}'))
    cursor = conn.cursor()
    # 提交并更改
    conn.commit()
    conn.close()


# 判断数据库是否存在表名
def whether_exist_table(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'Mult_Strategy_1_TimesNet_scaled_1'):
    '''
    功能: 新建一个数据库
    输入: 
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
        table_name: 表名
    Example:
    >>> whether_exist_table(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'Mult_Strategy_1_TimesNet_scaled_1')
    '''
    conn = sqlite3.connect(os.path.join(db_file_path, f'{db_name}'))
    cursor = conn.cursor()
    
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
    # 执行查询
    result = conn.execute(query).fetchall()
    # 提交并更改
    conn.commit()
    conn.close()
    return bool(result)


# DataFrame添加到自定义表名
def df_to_db(df, db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'csv_pred_temp'):
    '''
    功能: 将DataFrame添加到自定义的表名
    输入: 
        df: DataFrame
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
        table_name: 表名
    Example:
    >>> df_to_db(df, db_file_path="../daily_stock_daybar/", db_name = 'long_term_forecast', table_name = 'csv_pred_temp')
    '''
    # 连接到数据库
    conn = sqlite3.connect(os.path.join(db_file_path, f'{db_name}'))
    cursor = conn.cursor()

    # 判断表是否存在，如果不存在就创建，注意表的形状！
    # TEXT为文本字符， REAL为实数（包含int和float） ， NUMERIC(9,6) 指的是数值的长度，9是整个的长度，6是小数长度
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (tradingday TEXT, code TEXT, pred NUMERIC(9,6), primary key (tradingday,code));")

    # 将数据插入表中
    df.to_sql(table_name, conn, if_exists='append', index=False) # 如果数据表已存在有则往后新加

    # 提交更改并关闭连接
    conn.commit()
    conn.close()


# 从数据库读取表
def read_df_from_db(start_day, end_day, db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'csv_pred_temp'):
    '''
    功能: 从数据库读取表
    输入: 
        start_day: 起始日期
        end_day: 终止日期
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
        table_name: 表名
    输出:
        返回一个DataFrame
    Example:
    >>> read_df_from_db(start_day = None, end_day = None, db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'csv_pred_temp')
    '''
    # 连接到数据库
    conn = sqlite3.connect(os.path.join(db_file_path, f'{db_name}'))

    # 执行 SELECT 查询
    query = f"SELECT * FROM {table_name} WHERE tradingday BETWEEN '{start_day}' AND '{end_day}' ORDER BY tradingday, pred DESC;"
    results = pd.read_sql_query(query, conn)

    # 关闭连接
    conn.close()
    return results


# 从数据库中读取出已经存在的表名
def read_tabel_name_from_db(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast'):
    conn = sqlite3.connect(os.path.join(db_file_path, db_name))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return tables


# 删除数据库中的表
def drop_tabel(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'csv_pred_temp'):
    '''
    注意: 如果该表为空表则不能进行删除！
    功能: 删除数据库中的表
    输入: 
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
        table_name: 表名
    Example:
    >>> drop_tabel(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast', table_name = 'csv_pred_temp')
    '''
    # 连接到数据库
    conn = sqlite3.connect(os.path.join(db_file_path, f'{db_name}'))
    cursor = conn.cursor()

    # 执行删除表的操作
    cursor.execute(f'DROP TABLE IF EXISTS {table_name};')

    # 提交更改并关闭连接
    conn.commit()
    conn.close()

    print(f'{table_name} 数据表删除成功')


# 删除数据库
def drop_db(db_file_path = '../daily_stock_daybar/', db_name = 'long_term_forecast'):
    '''
    注意: 直接来一波删库跑路！!
    功能: 删除数据库
    输入: 
        db_file_path: 本地数据库的路径
        db_name: 数据库名字
    Example:
    >>> drop_db(db_file_path="../daily_stock_daybar/", table_name='csv_pred_temp')
    '''
    db_file = os.path.join(db_file_path, f'{db_name}')
    # 删除数据库文件
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f'{db_file} 删库成功！！！')
    else:
        print(f'{db_file} 数据库不存在')