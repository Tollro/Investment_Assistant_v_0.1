import akshare as ak
import pandas as pd
from typing import Literal

def get_stock_list():
    """获取A股上市公司列表"""
    try:
        stock_list = ak.stock_info_a_code_name()
        if stock_list is None or stock_list.empty:
            print("未获取到股票列表")
            return None
        return stock_list
    except Exception as e:
        print(f"获取股票列表时出错: {e}")
        return None

def get_stock_daily_data(symbol, start_date, end_date, adjust="qfq"):
    """使用AKShare获取股票日线数据，并确保数据类型和字段名与数据库匹配"""
    try:
        # 使用腾讯接口获取数据，symbol格式如 'sz000001' (平安银行)
        df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        if df is None or df.empty:
            print(f"未获取到股票 {symbol} 的数据")
            return None
        
        # 重命名列以匹配数据库表结构，并添加股票代码列
        df.rename(columns={
            'date': 'time',
            'amount': 'volume',
        }, inplace=True)
        df['symbol'] = symbol
        
        # 选择并排序需要的列，确保 'time' 是 datetime 类型
        df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        # 剔除完全重复的行
        df = df.drop_duplicates()
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"获取股票 {symbol} 数据时出错: {e}")
        return None

# def get_stock_data(symbol, timescale:Literal["daily", "weekly", "monthly"], start_date, end_date, adjust="qfq"):
#     try:

#         if timescale=="daily":
#             df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust=adjust)
#         elif timescale=="weekly":
#             df = ak.stock_zh_a_hist(symbol=symbol, period="weekly", start_date=start_date, end_date=end_date, adjust=adjust)
#         elif timescale=="monthly":
#             df = ak.stock_zh_a_hist(symbol=symbol, period="monthly", start_date=start_date, end_date=end_date, adjust=adjust)
#         else:
#             print(f"time参数： {timescale} 不符合格式！")
#             return None  
        
#         if df is None or df.empty:
#             print(f"未获取到股票 {symbol} 的数据")
#             return None
        
#         # 重命名列以匹配数据库表结构，并添加股票代码列
#         df.rename(columns={
#             'date': 'time',
#             'amount': 'volume',
#         }, inplace=True)
#         df['symbol'] = symbol
        
#         # 选择并排序需要的列，确保 'time' 是 datetime 类型
#         df = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
#         df['time'] = pd.to_datetime(df['time'])
#         return df
#     except Exception as e:
#         print(f"获取股票 {symbol} 数据时出错: {e}")
#         return None

# def get_fundamental_data(symbol):
#     """
#     获取A股上市公司基本面数据（财务指标）
#     接口: stock_financial_analysis_indicator
#     描述: 获取新浪财经-财务分析-财务指标
#     """
#     try:
#         # 提取纯数字代码（兼容 'sz000001', '000001', '000001.SZ' 等格式）
#         code = symbol.replace('sz', '').replace('sh', '').replace('SZ', '').replace('SH', '')
#         if '.' in code:
#             code = code.split('.')[0]

#         df = ak.stock_financial_analysis_indicator(symbol=code)
#         if df is None or df.empty:
#             print(f"未获取到股票 {symbol} 的基本面数据")
#             return None
#         df['symbol'] = symbol
#         return df
#     except Exception as e:
#         print(f"获取股票 {symbol} 基本面数据时出错: {e}")
#         return None

# 获取财务摘要
def get_fundamental_data(symbol):
    """
    获取A股上市公司基本面数据（财务摘要）
    :param symbol: 股票代码，如 'sz000002' 或 '000002'
    :return: pandas.DataFrame 或 None
    """
    try:
        # 1. 从输入中提取纯数字代码 (如 '000002')
        code = symbol.replace('sz', '').replace('sh', '').replace('SZ', '').replace('SH', '')
        if '.' in code:
            code = code.split('.')[0]
        
        # 2. 打印信息，便于调试
        print(f"正在为 {symbol} (处理后的代码: {code}) 获取财务摘要...")

        # 3. 使用更稳定的 stock_financial_abstract 接口
        df = ak.stock_financial_abstract(symbol=code)
        
        # 4. 检查是否获取到数据
        if df is None or df.empty:
            print(f"未获取到股票 {symbol} 的财务摘要数据。")
            return None
        
        # 5. 添加股票代码列，方便后续处理
        df['symbol'] = symbol
        print(f"成功获取到 {symbol} 的 {len(df)} 条财务摘要记录。")
        return df

    except Exception as e:
        print(f"获取股票 {symbol} 基本面数据时出错: {e}")
        return None

# 获取完整的财务报表
def get_financial_report(symbol, report_type=Literal["资产负债表", "利润表", "现金流量表"]):
    """
    获取A股上市公司完整财务报表
    :param symbol: 股票代码，如 'sz000002' 或 '000002'
    :param report_type: 报表类型，可选 "资产负债表", "利润表", "现金流量表"
    :return: pandas.DataFrame
    """
    try:
        # 确保代码带市场前缀（新浪接口要求）
        if not symbol.startswith(('sz', 'sh', 'SZ', 'SH')):
            if symbol.startswith('6'):
                prefix = 'sh'
            else:
                prefix = 'sz'
            code = prefix + symbol
        else:
            code = symbol.lower()
        
        df = ak.stock_financial_report_sina(stock=code, symbol=report_type)
        if df is None or df.empty:
            print(f"未获取到股票 {symbol} 的{report_type}")
            return None
        
        df = df.copy()
        df['symbol'] = symbol
        df['report_type'] = report_type
        print(f"成功获取 {symbol} 的{report_type}，共 {len(df)} 行")
        return df
    except Exception as e:
        print(f"获取股票 {symbol} 的{report_type}时出错: {e}")
        return None

def get_news_titles(symbol, top_n=20):
    """
    获取A股个股的新闻标题和内容摘要
    接口: stock_news_em
    描述: 获取东方财富指定个股的新闻资讯数据，最多返回当日最近top_n条
    """
    try:
        # 提取纯数字代码
        code = symbol.replace('sz', '').replace('sh', '').replace('SZ', '').replace('SH', '')
        if '.' in code:
            code = code.split('.')[0]
            
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            print(f"未获取到股票 {symbol} 的新闻数据")
            return None
        
        # 中英文列名映射表（覆盖常见变体）
        column_map = {
            '新闻标题': 'title',
            '标题': 'title',
            '新闻内容': 'content',
            '内容': 'content',
            '发布时间': 'public_time',
            '时间': 'public_time',
            '新闻链接': 'url',
            '链接': 'url',
            '关键词': 'keywords',
        }
        # 仅重命名存在的列
        df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

        # 添加代码列，保留需要的字段
        df['symbol'] = symbol

        # 筛选需要的列（若存在则保留）
        wanted_cols = ['symbol', 'title', 'content', 'public_time', 'url']
        available_cols = [col for col in wanted_cols if col in df.columns]
        if not available_cols:
            print(f"警告：新闻数据列名不匹配，实际列为: {df.columns.tolist()}")
            return df.head(top_n)

        # 返回前top_n条记录，保留标题、内容、发布时间、URL等关键字段
        df = df[['symbol', 'title', 'content', 'public_time', 'url']].head(top_n)
        return df
    except Exception as e:
        print(f"获取股票 {symbol} 新闻数据时出错: {e}")
        return None

# --- 主程序：演示如何获取平安银行（000001）的数据 ---
if __name__ == "__main__":
    # # 1. 获取股票代码
    # stock_code = get_stock_code("平安银行")
    # if stock_code:
    #     print(f"平安银行的代码: {stock_code}")
    #     # 2. 获取股票数据
    #     stock_df = get_stock_data(symbol=stock_code, timescale="daily", start_date="20230101", end_date="20231231", adjust="qfq")
    #     if stock_df is not None:
    #         print(stock_df.head())
    #     else:
    #         print("未能获取平安银行数据")
    # else:
    #     print("未能获取平安银行代码")
    
    # print("获取股票列表:")
    # stock_list = get_stock_list()
    # if stock_list is not None:
    #     print(stock_list.head())
    # else:
    #     print("未能获取股票列表")

    """检测代码"""
    # stock_df = get_stock_daily_data(symbol="sz000001", timescale="daily", start_date="20230101", end_date="20230231", adjust="qfq")
    # if stock_df is not None:
    #     print(stock_df.head())
    # else:
    #     print("未能获取平安银行数据")

    # newstitles = get_news_titles("000002")
    # print("newstitles:",newstitles)

    financial_data = get_financial_report("000002", report_type="现金流量表")
    print("financial_data:",financial_data)