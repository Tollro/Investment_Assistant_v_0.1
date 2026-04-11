import akshare as ak
import pandas as pd

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

def get_stock_code(name):
    """根据股票名称获取股票代码"""
    try:
        stock_list = get_stock_list()
        if stock_list is not None:
            # 查找名称匹配的股票（不区分大小写）
            match = stock_list[stock_list['name'].str.contains(name, case=False, na=False)]
            if not match.empty:
                # 返回第一个匹配的代码，并添加市场前缀
                code = match.iloc[0]['code']
                if code.startswith('6'):
                    return f"sh{code}"
                else:
                    return f"sz{code}"
            else:
                print(f"未找到股票名称: {name}")
                return None
        return None
    except Exception as e:
        print(f"获取股票代码时出错: {e}")
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
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"获取股票 {symbol} 数据时出错: {e}")
        return None

# --- 主程序：演示如何获取平安银行（000001）的数据 ---
if __name__ == "__main__":
    # 1. 获取股票代码
    stock_code = get_stock_code("平安银行")
    if stock_code:
        print(f"平安银行的代码: {stock_code}")
        # 2. 获取股票数据
        stock_df = get_stock_data(symbol=stock_code, start_date="20230101", end_date="20231231", adjust="qfq")
        if stock_df is not None:
            print(stock_df.head())
        else:
            print("未能获取平安银行数据")
    else:
        print("未能获取平安银行代码")
    
    print("获取股票列表:")
    stock_list = get_stock_list()
    if stock_list is not None:
        print(stock_list.head())
    else:
        print("未能获取股票列表")
