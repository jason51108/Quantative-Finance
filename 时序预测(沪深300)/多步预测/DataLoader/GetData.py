import pandas as pd
from sqlalchemy import create_engine


# 获取给定日期前或后n天的所有交易日期
def Get_All_Tradingday(date:str, n:int, forward=False, engine = create_engine("mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server"))->list:
    """
    功能: 获取给定日期前或后n天的所有交易日期
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        date: 字符串,代表日期,如'20210101'
        n: int,代表date日期前或后所需要的交易日期天数
        forward: 默认为False,代表日期往后
    输出: 
        [time_1,time_2,.....],其中time_i为Timestamp日期格式

    Example:
    >>> Get_All_Tradingday( date='20220101',n=3,forward=False)

    返回: [Timestamp('2022-01-04 00:00:00'),Timestamp('2022-01-05 00:00:00'),Timestamp('2022-01-06 00:00:00')]
    """

    if forward:
        query_pre_end_tradingday = f"""SELECT TOP {n} * FROM daily..tradingday WHERE tradingday < '{date}' ORDER BY tradingday DESC"""
        start_date = pd.read_sql(query_pre_end_tradingday, engine).tradingday.tolist()[-1]
        query_trad_tradingday = f"SELECT distinct tradingday FROM daily..tradingday WHERE  tradingday >= '{start_date}' AND tradingday <= '{date}'"  
        tradingday_list = sorted(list(set(pd.to_datetime(pd.read_sql(query_trad_tradingday, engine).tradingday).tolist())))
    else:
        query_pre_end_tradingday = f"""SELECT TOP {n} * FROM daily..tradingday WHERE tradingday > '{date}' ORDER BY tradingday"""
        end_date = pd.read_sql(query_pre_end_tradingday, engine).tradingday.tolist()[-1]
        query_trad_tradingday = f"SELECT distinct tradingday FROM daily..tradingday WHERE  tradingday >= '{date}' AND tradingday <= '{end_date}'"  
        tradingday_list = sorted(list(set(pd.to_datetime(pd.read_sql(query_trad_tradingday, engine).tradingday).tolist())))
    return tradingday_list


# 获取给定起始时间和结束时间的交易日期
def Get_All_Tradingday_1(beginday:str, endday:str, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    """
    功能: 获取给定起始时间和结束时间的交易日期
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,代表起始日期,如'20210101'
        endday: 字符串,代表结束日期,如'20230101'
    输出: 
        [time_1,time_2,.....],其中time_i为Timestamp日期格式

    Example:
    >>> Get_All_Tradingday_1( beginday='20220101',endday='20230101')

    返回: [Timestamp('2022-01-04 00:00:00'),Timestamp('2022-01-05 00:00:00'),Timestamp('2022-01-06 00:00:00'),......]
    """
    query_trad_tradingday = f"SELECT distinct tradingday FROM daily..tradingday WHERE  tradingday >= '{beginday}' AND tradingday < '{endday}'" 
    tradingday_list = sorted(list(set(pd.to_datetime(pd.read_sql(query_trad_tradingday, engine).tradingday).tolist())))
    return tradingday_list


# 获取起始日期到终止日期的各行业名称
def Get_stock_industry(beginday:str, endday:str, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server'))->list:
    """
    功能: 获取起始日期到终止日期的各行业名称
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,代表起始日期,如'20210101'
        endday: 字符串,代表结束日期,如'20230101'
    输出:
        ['交运设备','交通运输',......]

    Example:
    >>> Get_stock_industry( beginday='20220101',endday='20230101')

    返回: ['交运设备','交通运输',......]
    """
    query_stock_df = f"""SELECT DISTINCT(b.industry)
                         FROM daily..daybar a 
                         JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                         WHERE datediff(day,listeddate,a.tradingday) > 180 AND tradeable = 1 AND volume > 0 --剔除ST和次新股
                         AND a.tradingday BETWEEN '{beginday}' AND '{endday}'"""
    df = pd.read_sql(query_stock_df, engine)
    industry_list = sorted(list(set(df.industry.tolist())))
    return industry_list


# 获取指数
def Get_index(beginday, endday, stock_code:str, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    '''
    功能: 获取起始日期到终止日期的单只股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,代表起始日期,如'20210101'
        endday: 字符串,代表结束日期,如'20230101'
    输出:
        某只股票的DataFrame
    
    Example:
    >>> Get_stock(beginday='20220101',endday='20230101',stock_code='SH600862')

    返回: 一个DataFrame
    '''
    query_stock_df = f'''SELECT * FROM dayindex WHERE code IN ('{stock_code}') AND tradingday BETWEEN {beginday} AND {endday} AND volume > 0'''
    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    df = df.set_index(['code','tradingday'])
    df.sort_index(inplace=True)
    return_ = ((df['close'] / df['pre_close'])-1)
    df['return'] = return_
    df.drop(['pre_close', 'factor', 'turnover'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df


# 获取行业内所有股票代码
def Get_industry_stock(beginday, endday, industry, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    """
    功能: 获取起始日期到终止日期的某行业的所有股票代码
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,代表起始日期,如'20210101'
        endday: 字符串,代表结束日期,如'20230101'
        industry: 字符串,如'交通运输'
    输出:
        ['SH600862', 'SH600967', 'SZ000519', 'SZ002190']
    
    Example:
    >>> Get_industry_stock(beginday='20220101',endday='20230101',industry='交运设备')

    返回: ['SH600862', 'SH600967', 'SZ000519', 'SZ002190']
    """
    query_stock_df = f"""SELECT DISTINCT(a.code)
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE datediff(day,listeddate,a.tradingday) > 180 AND tradeable = 1 AND volume > 0 AND b.industry='{industry}'--剔除ST和次新股
                        AND a.tradingday BETWEEN '{beginday}' AND '{endday}'"""
    
    df = pd.read_sql(query_stock_df, engine)
    stock_list = sorted(list(set(df.code.tolist())))
    return stock_list


# 获得单个股票信息
def Get_stock(beginday, endday, stock_code:str, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    '''
    功能: 获取起始日期到终止日期的单只股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,代表起始日期,如'20210101'
        endday: 字符串,代表结束日期,如'20230101'
        industry: 字符串,如'SH600862'
    输出:
        某只股票的DataFrame
    
    Example:
    >>> Get_stock(beginday='20220101',endday='20230101',stock_code='SH600862')

    返回: 一个DataFrame
    '''
    query_stock_df = f'''SELECT * FROM daybar WHERE code IN ('{stock_code}') AND tradingday BETWEEN {beginday} AND {endday} AND volume > 0'''
    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    df = df.set_index(['code','tradingday'])
    df.sort_index(inplace=True)
    return_ = ((df['close'] / df['pre_close'])-1)
    df['return'] = return_
    df.drop(['pre_close', 'factor', 'turnover'], axis=1, inplace=True)
    df.dropna(inplace=True)
    # df.rename(columns={'tradingday': 'date'}, inplace=True)
    return df

# 获取某个行业的股票信息
def Get_stock_info(beginday, endday, industry:str, data_type='Timeseries', engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    """
    功能: 获取起始日期到终止日期的某个行业的股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,如'20210101'
        endday: 字符串,如'20220101'
        industry: 字符串,如'交通运输'
        data_type: 字符串,如'Timeseries'时序数据，'Panel'面板数据。
    输出:
        某个行业股票的DataFrame
    注意：面板数据和时序数据的差别在于DataFrame的Index的顺序问题
    Example:
    >>> Get_stock_info(beginday='20220101',endday='20230101',industry='交通运输')

    返回: 一个DataFrame
    """
    query_stock_df = f'''WITH raw AS (
                        SELECT a.tradingday,a.code,[open],high,low,[close],pre_close,volume,turnover,b.industry,b.totalMV,b.negotiableMV,
                        volume_chg=CASE WHEN lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE volume*1.0/lag(volume,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        turnover_chg=CASE WHEN lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE turnover*1.0/lag(turnover,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        totalMV_chg=CASE WHEN lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE totalMV*1.0/lag(totalMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END,
                        negotiableMV_chg=CASE WHEN lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) <= 0 THEN 1 ELSE negotiableMV*1.0/lag(negotiableMV,1,NULL) OVER (ORDER BY a.code,a.tradingday) END
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE a.code IN (
                        SELECT code FROM (SELECT tradingday,code,tradeable=CASE WHEN datediff(day,listeddate,tradingday) > 180 THEN 1 ELSE 0 END&tradeable 
                        FROM daily..stockinfo WHERE tradingday BETWEEN '{beginday}' AND '{endday}') x GROUP BY code HAVING count(*) = sum(tradeable))	-- 剔除次新股和ST股票
                        AND a.tradingday BETWEEN dateadd(month,-1,'{beginday}') AND '{endday}'
                        )
                        SELECT a.tradingday,a.code,

                        ------价格四选一----
                        -- 原始价格
                        [open],high,low,[close],pre_close,
                        /*
                        --价格转比例
                        [open]=[open]/pre_close,high=[high]/pre_close,low=[low]/pre_close,[close]=[close]/pre_close,

                        --原始价格去极值
                        [open]=CASE WHEN [open]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [open]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [open] END,
                        high=CASE WHEN [high]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [high]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [high] END,
                        low=CASE WHEN [low]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [low]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [low] END,
                        [close]=CASE WHEN [close]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [close]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [close] END,


                        --价格转比例去极值
                        [open]=CASE WHEN [open]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [open]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [open]/pre_close END,
                        high=CASE WHEN [high]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [high]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [high]/pre_close END,
                        low=CASE WHEN [low]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [low]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [low]/pre_close END,
                        [close]=CASE WHEN [close]/pre_close > 1.1 THEN pre_close * 1.1 WHEN [close]/pre_close < 0.9 THEN pre_close * 0.9 ELSE [close]/pre_close END,
                        -------end------
                        */
                        ------量额四选一----
                        --原始量额
                        volume,
                        /*
                        --量额转比例
                        volume=volume_chg,
                        turnover=turnover_chg,

                        --原始量额去极值
                        volume = CASE WHEN volume_chg = 0 THEN 0 WHEN volume_chg > 10 THEN volume/volume_chg*10 WHEN volume_chg < 0.1 THEN volume/volume_chg*0.1 ELSE volume END,
                        turnover = CASE WHEN turnover_chg = 0 THEN 0 WHEN turnover_chg > 10 THEN turnover/turnover_chg*10 WHEN turnover_chg < 0.1 THEN turnover/turnover_chg*0.1 ELSE turnover END,

                        --量额转比例去极值
                        volume = CASE WHEN volume_chg = 0 THEN 1 WHEN volume_chg > 10 THEN 10 WHEN volume_chg < 0.1 THEN 0.1 ELSE volume_chg END,
                        turnover = CASE WHEN turnover_chg = 0 THEN 1 WHEN turnover_chg > 10 THEN 10 WHEN turnover_chg < 0.1 THEN 0.1 ELSE turnover_chg END,
                        -------end------
                        */
                        a.industry
                        FROM raw a
                        LEFT JOIN (
                            SELECT tradingday,industry,industry_open=avg([open]),industry_high=avg(high),industry_low=avg(low),industry_close=avg([close]) FROM raw GROUP BY tradingday,industry
                        ) b ON a.tradingday = b.tradingday AND a.industry = b.industry
                        WHERE a.tradingday BETWEEN '{beginday}' AND '{endday}' 
                        AND volume > 0
                        ORDER BY a.code,a.tradingday'''

    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    if data_type == 'Timeseries':
        df = df.set_index(['code', 'tradingday'])
    if data_type == 'Panel':
        df = df.set_index(['tradingday', 'code'])
    df.sort_index(inplace=True)
    # 如果行业存在，则筛选出industry为行业的信息，否则表示所有
    if industry is not None:
        df = df[df['industry'] == industry]
    return_ = (((df['close'] / df['pre_close'])-1))
    df['return'] = return_
    df.drop(['pre_close', 'industry'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

# 获取某个行业的股票信息(按照股票code去筛选)
def Get_stock_info_1(beginday, endday, industry:str, data_type='Timeseries', engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    """
    功能: 获取起始日期到终止日期的某个行业的股票信息
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        beginday: 字符串,如'20210101'
        endday: 字符串,如'20220101'
        industry: 字符串,如'交通运输'
        data_type: 字符串,如'Timeseries'时序数据，'Panel'面板数据。
    输出:
        某个行业股票的DataFrame
    注意：面板数据和时序数据的差别在于DataFrame的Index的顺序问题
    Example:
    >>> Get_stock_info_1(beginday='20220101',endday='20230101',industry='交通运输')

    返回: 一个DataFrame
    """
    query_stock_df = f'''SELECT a.tradingday, a.code, [open], high, low, [close], volume, pre_close
                        FROM daily..daybar a 
                        JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                        WHERE DATEDIFF(day, listeddate, a.tradingday) > 180 AND tradeable = 1 AND volume > 0 AND a.tradingday BETWEEN {beginday} AND {endday} 
                        AND a.code IN (SELECT DISTINCT(code)
                        FROM daily..stockinfo
                        WHERE tradingday = (SELECT TOP 1 * FROM daily..tradingday WHERE tradingday <= '{endday}' ORDER BY tradingday DESC) AND industry = '{industry}')
                        ORDER BY code, tradingday;
                        '''

    df = pd.read_sql(query_stock_df, engine)
    df.tradingday = pd.to_datetime(df.tradingday)
    if data_type == 'Timeseries':
        df = df.set_index(['code', 'tradingday'])
    if data_type == 'Panel':
        df = df.set_index(['tradingday', 'code'])
    df.sort_index(inplace=True)
    # if industry is not None:
    #     df = df[df['industry'] == industry]
    return_ = (((df['close'] / df['pre_close'])-1))
    df['return'] = return_
    df.drop(['pre_close'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

# 获取小市值股票
def Get_data_miniMV(day, n, forward=False, engine = create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    # 根据forward进行向前或者向后查询
    if forward == False:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday < '{day}' ORDER BY tradingday DESC'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[-1]
        end_day = gap_days[0]
        # 查询
        quary = f"""SELECT a.tradingday,a.code,
                    [open],high,low,[close],
                    volume,
                    -- b.totalMV,b.negotiableMV,
                    label=log([close]/pre_close)
                    FROM daily..daybar a 
                    JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                    WHERE datediff(day,listeddate,a.tradingday) > 180 AND tradeable = 1 AND volume > 0 AND b.totalMV < 5000000000 --剔除ST和次新股
                    AND a.tradingday BETWEEN '{start_day}' AND '{end_day}'
                    ORDER BY code,tradingday
                """
        df = pd.read_sql(quary, engine)
        df.rename(columns={'label':'return'},inplace=True)
        df.set_index(['code', 'tradingday'], inplace=True)
        return df
    else:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday > '{day}' ORDER BY tradingday'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[0]
        end_day = gap_days[-1]
        # 查询
        quary = f"""SELECT a.tradingday,a.code,
                    [open],high,low,[close],
                    volume,
                    -- b.totalMV,b.negotiableMV,
                    label=log([close]/pre_close)
                    FROM daily..daybar a 
                    JOIN daily..stockinfo b ON a.tradingday = b.tradingday AND a.code = b.code
                    WHERE datediff(day,listeddate,a.tradingday) > 180 AND tradeable = 1 AND volume > 0 AND b.totalMV < 3000000000 --剔除ST和次新股
                    AND a.tradingday BETWEEN {start_day} AND {end_day}
                    ORDER BY code,tradingday
                """
        df = pd.read_sql(quary, engine)
        df.rename(columns={'label':'return'},inplace=True)
        df.set_index(['code', 'tradingday'], inplace=True)
        return df

# 获取给定日期前或后多少个交易日
def cheak_day(day,n,forward=False, engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    if forward == False:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday < '{day}' ORDER BY tradingday DESC'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[-1]
        end_day = gap_days[0]
    else:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday > '{day}' ORDER BY tradingday'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[0]
        end_day = gap_days[-1]
    return start_day,end_day

# 读取数据合集(向前取数据没问题，向后取可能有点问题)
def Get_data(day, n, data_type,forward=False,
             industry=None,
             stock=None,
             Ashares=False,
             engine=create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')):
    """
    功能: 获取指定日期前或后N天的的股票数据读取合集
    输入: 
        engine: 默认为"create_engine('mssql+pyodbc://visitor:123@192.168.5.18/daily?driver=SQL+Server')"
        day: 字符串,如'20230101'
        n: 选取日期的长度,从day之前或者之后选择n天
        data_type: 字符串,如'Timeseries'时序数据，'Panel'面板数据
        forward:默认为False,代表日期往后
    输出:
        某个行业或者单只股票或者全部的股票的DataFrame
    注意：在进行参数传递过程中，industry、stock和Ashares只需要传入一个就好，其余两个无须进行传递
    Example 1:
    >>> Get_data(day='20220101',n=10,forward=False,industry='交通运输')

    返回: 一个交通运输行业的DataFrame

    Example 2:
    >>> Get_data(day='20220101',n=10,Ashares=True)

    返回: 一个所有行业的DataFrame
    """

    # 用来判断industry、stock和Ashares三个参数是否只有一个成立
    a = True if industry else False
    b = True if stock else False
    assert (a + b + bool(Ashares)) == 1, "注意：在进行参数传递过程中，industry、stock和Ashares有且只能传入一个参数，其余两个无须进行传递，例如Get_data( endday='20220101',train_len=15,iag_len=10,industry='交通运输')"

    # 根据forward进行向前或者向后查询
    if forward == False:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday < '{day}' ORDER BY tradingday DESC'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[-1]
        end_day = gap_days[0]
        # 如果行业非空，获取某个行业的股票信息
        if industry is not None:
            data = Get_stock_info_1(start_day, end_day, industry)

        # 如果行业为空，股票代码不为空，则获取某个股票代码的信息
        if industry is None and stock is not None:
            data = Get_stock(start_day, end_day, stock_code=stock)

        # 如果行业为空，股票代码也为空，则返回空(这里存在是没有意义的)
        if industry is None and stock is None:
            data = Get_stock_info(start_day, end_day, industry)

        # 如果Ashares打开则代表所有行业的信息
        if Ashares:
            data = Get_stock_info(start_day, end_day, industry=None, data_type=data_type)  #Get_stock_info这个函数如果未指定industry的话则代表所有行业
        return data

    else:
        require_start_tradingday = f'''SELECT TOP {n} * FROM daily..tradingday WHERE tradingday > '{day}' ORDER BY tradingday'''
        gap_days = pd.read_sql(require_start_tradingday, engine).tradingday.tolist()
        start_day = gap_days[0]
        end_day = gap_days[-1]
        # 如果行业非空，获取某个行业的股票信息
        if industry is not None:
            data = Get_stock_info(day, end_day, industry)

        # 如果行业为空，股票代码不为空，则获取某个股票代码的信息
        if industry is None and stock is not None:
            data = Get_stock(  day, end_day, stock_code=stock)

        # 如果行业为空，股票代码也为空，则返回空(这里存在是没有意义的)
        if industry is None and stock is None:
            data = Get_stock_info(day, end_day, industry)

        # 如果Ashares打开则代表所有行业的信息
        if Ashares:
            data = Get_stock_info(day, end_day, industry=None, data_type=data_type)  #Get_stock_info这个函数如果未指定industry的话则代表所有行业
        return data
