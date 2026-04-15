import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
import sqlite3
warnings.filterwarnings('ignore')
DATABASE_FILE = './db/stock_list.db'

def normalize_stock_code(code):
    """将无前缀股票代码转换为带前缀的代码(sh/ sz)。"""
    if not code:
        return None
    code = str(code).strip()
    if code.startswith(('sh', 'sz')):
        return code
    if code.startswith('6'):
        return f"sh{code}"
    return f"sz{code}"

def query_by_code(code):
    normalized_code = normalize_stock_code(code)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT code, name FROM stock_catalog WHERE code = ?', (normalized_code,))
    row = cursor.fetchone()
    conn.close()
    return row


def query_by_name_keyword(keyword):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # 使用 LIKE 进行模糊匹配（% 表示任意字符）
    cursor.execute('SELECT code, name FROM stock_catalog WHERE name LIKE ?', (f'%{keyword}%',))
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_all_data(symbol: str, start_date="20241201", end_date="20250101"):
    """
    获取股票的全面数据，包括实时行情、日线/周线/月线K线、财务数据、行业平均、指数数据等
    
    Parameters
    ----------
    symbol : str
        股票代码，格式如 "sh600519", "sz000001"
    top_n : int
        最近几日的日线数据条数，默认30
        周线和月线数据会根据日线数据自动转换，也会返回相应周期的最近数据
    
    Returns
    -------
    dict
        包含股票全面数据的字典
    """
    result = {
        "stock_code": symbol,
        "company_name": "",
        "market_data": {
            "realtime": {},
            "kline_daily": [],
            "kline_weekly": [],
            "kline_monthly": []
        },
        "financial_reports": {
            "balance_sheet": [],
            "income_statement": [],
            "cash_flow": [],
            "report_date": ""
        },
        "industry_avg": {
            "pe": None,
            "pb": None,
            "roe": None
        },
        "index_data": {
            "sh000001": {"price": None, "change_pct": None},
            "sz399001": {"price": None, "change_pct": None},
            "sz399006": {"price": None, "change_pct": None}
        }
    }
    
    # 获取实时数据
    result = get_realtime_with_name(symbol, result)
    
    # 2. 获取日线数据并转换为周线和月线
    result = get_stock_kline_data(symbol, result, start_date, end_date)
    
    # 3. 获取财务报告
    result = get_financial_report(symbol, result)
    
    # 4. 计算财务比率
    result = compute_financial_ratios(symbol, result)
    
    # 5. 计算技术指标
    result = compute_technical_indicators(result)
    
    # 6. 获取行业平均估值
    result = get_industry_avg(result)
    
    # 7. 获取指数数据
    result = get_index_data(result)
    
    return result


def get_realtime_with_name(symbol: str, result: dict) -> dict:
    """
    获取实时行情数据和公司名称，并填充到 result 中。

    Parameters
    ----------
    symbol : str
        股票代码，如 "sh600519"、"600519" 或 "sz000001"
    result : dict
        结果字典，需包含 market_data.realtime 字段

    Returns
    -------
    dict
        更新后的 result
    """
    try:
        # 获取沪深A股实时行情（全市场）
        spot_df = ak.stock_zh_a_spot()

        if spot_df is None or spot_df.empty:
            print("警告: 未获取到全市场实时行情数据")
            return result

        # ---- 兼容不同 symbol 格式 ----
        # 常见格式: "sh600519" -> 需要 "600519"；"000001" -> 需要 "000001"
        # AKShare 返回的 '代码' 列一般为纯数字（6位）
        pure_code = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
        if len(pure_code) != 6:
            pure_code = symbol  # 若不符合预期，直接用原 symbol 尝试

        # 筛选目标股票（同时考虑可能带市场前缀的情况）
        stock_spot = spot_df[
            (spot_df['代码'] == pure_code) | (spot_df['代码'] == symbol)
        ]

        if not stock_spot.empty:
            row = stock_spot.iloc[0]

            # ---- 字段映射（兼容中英文列名） ----
            # 可能的列名：名称/股票名称，最新价/最新，涨跌幅，成交量/成交额，最高，最低
            col_map = {
                '名称': ['名称', '股票名称', 'name'],
                '最新价': ['最新价', '最新', 'price'],
                '涨跌幅': ['涨跌幅', '涨跌', 'change_pct'],
                '成交量': ['成交量', 'volume'],
                '最高': ['最高', 'high'],
                '最低': ['最低', 'low']
            }

            def get_val(col_key):
                for candidate in col_map[col_key]:
                    if candidate in row.index:
                        return row[candidate]
                return None

            # 填充公司名称
            name_val = get_val('名称')
            result["company_name"] = str(name_val) if name_val is not None else ""

            # 填充实时行情
            realtime = {
                "price": None,
                "change_pct": None,
                "volume": None,
                "high": None,
                "low": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            try:
                realtime["price"] = float(get_val('最新价')) if get_val('最新价') is not None else None
            except:
                pass
            try:
                realtime["change_pct"] = float(get_val('涨跌幅')) if get_val('涨跌幅') is not None else None
            except:
                pass
            try:
                realtime["volume"] = int(float(get_val('成交量'))) if get_val('成交量') is not None else None
            except:
                pass
            try:
                realtime["high"] = float(get_val('最高')) if get_val('最高') is not None else None
            except:
                pass
            try:
                realtime["low"] = float(get_val('最低')) if get_val('最低') is not None else None
            except:
                pass

            result["market_data"]["realtime"] = realtime

        else:
            # 未找到对应股票时，仍填充空结构
            result["market_data"]["realtime"] = {
                "price": None,
                "change_pct": None,
                "volume": None,
                "high": None,
                "low": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"警告: 未找到股票 {symbol} 的实时行情数据（尝试代码: {pure_code}）")

    except Exception as e:
        print(f"获取实时行情失败: {e}")
        # 异常情况下也保证结构存在
        result["market_data"]["realtime"] = {
            "price": None,
            "change_pct": None,
            "volume": None,
            "high": None,
            "low": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    return result


def get_stock_kline_data(symbol: str,
                         result: dict,
                         start_date: str = "202401221",
                         end_date: str = "20250101") -> dict:
    """
    获取股票日线数据，并转换为周线和月线K线，填充到 result 的 market_data 中。

    Args:
        symbol: 股票代码，如 "sh600519" 或 "600519"
        result: 待填充的结果字典，需包含 market_data 结构
        start_date: 起始日期，格式 "YYYYMMDD"
        end_date: 结束日期，格式 "YYYYMMDD"
    
    Returns:
        填充后的 result 字典
    """
    try:
        # 1. 获取日线数据（前复权）
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df is None or df.empty:
            print(f"警告: 未获取到 {symbol} 的日线数据")
            return result

        # 2. 统一列名（兼容中文和英文列名）
        rename_dict = {}
        # 日期列
        for col in df.columns:
            if col in ['日期', 'date']:
                rename_dict[col] = 'date'
            elif col in ['开盘', 'open']:
                rename_dict[col] = 'open'
            elif col in ['最高', 'high']:
                rename_dict[col] = 'high'
            elif col in ['最低', 'low']:
                rename_dict[col] = 'low'
            elif col in ['收盘', 'close']:
                rename_dict[col] = 'close'
            elif col in ['成交量', 'volume', 'amount']:  # 注意 amount 有时是成交额，需确认
                rename_dict[col] = 'volume'
        df.rename(columns=rename_dict, inplace=True)

        # 确保必需列存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"数据缺少必要列: {missing}，现有列: {df.columns.tolist()}")

        # 3. 数据类型转换与索引设置
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # 4. 生成日线列表
        daily_data = []
        for idx, row in df.iterrows():
            daily_data.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        result["market_data"]["kline_daily"] = daily_data

        # 转换为周线
        result = resample_to_weekly(df, result)

        # 转换为月线
        result = resample_to_monthly(df, result)


    except Exception as e:
        print(f"获取日线数据失败: {e}")

    return result


def resample_to_weekly(df: pd.DataFrame, result: dict) -> dict:
    """
    将日线DataFrame转换为周线，写入result的market_data中（返回全部历史周线）

    Parameters
    ----------
    df : pd.DataFrame
        日线数据，必须包含列: date(索引或列), open, high, low, close, volume
    result : dict
        目标结果字典

    Returns
    -------
    dict
        更新后的result
    """
    try:
        df_copy = df.copy()
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy.set_index('date', inplace=True)
        elif df_copy.index.name != 'date':
            df_copy.index = pd.to_datetime(df_copy.index)

        weekly = df_copy.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        weekly_data = []
        for idx, row in weekly.iterrows():  # 遍历全部周线
            weekly_data.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        result["market_data"]["kline_weekly"] = weekly_data

    except Exception as e:
        print(f"转换为周线失败: {e}")

    return result


def resample_to_monthly(df: pd.DataFrame, result: dict) -> dict:
    """
    将日线DataFrame转换为月线，写入result的market_data中（返回全部历史月线）

    Parameters
    ----------
    df : pd.DataFrame
        日线数据，必须包含列: date(索引或列), open, high, low, close, volume
    result : dict
        目标结果字典

    Returns
    -------
    dict
        更新后的result
    """
    try:
        df_copy = df.copy()
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy.set_index('date', inplace=True)
        elif df_copy.index.name != 'date':
            df_copy.index = pd.to_datetime(df_copy.index)

        monthly = df_copy.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        monthly_data = []
        for idx, row in monthly.iterrows():  # 遍历全部月线
            monthly_data.append({
                "time": idx.strftime("%Y-%m-%d"),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row['volume'])
            })
        result["market_data"]["kline_monthly"] = monthly_data

    except Exception as e:
        print(f"转换为月线失败: {e}")

    return result


def get_financial_report(symbol: str, result: dict) -> dict:
    """
    获取财务报告（资产负债表、利润表、现金流量表）
    
    Parameters
    ----------
    symbol : str
        股票代码
    result : dict
        结果字典
    
    Returns
    -------
    dict
        更新后的结果字典
    """
    try:
        # 资产负债表
        balance_df = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
        if balance_df is not None and not balance_df.empty:
            latest_balance = balance_df.iloc[0].to_dict()
            result["financial_reports"]["balance_sheet"] = [
                {"报表日期": latest_balance.get("报告日", ""),
                 "资产总计": latest_balance.get("资产总计", ""),
                 "负债合计": latest_balance.get("负债合计", "")}
            ]
        
        # 利润表
        income_df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
        if income_df is not None and not income_df.empty:
            latest_income = income_df.iloc[0].to_dict()
            result["financial_reports"]["income_statement"] = [
                {"报表日期": latest_income.get("报告日", ""),
                 "营业收入": latest_income.get("营业总收入", ""),
                 "净利润": latest_income.get("净利润", "")}
            ]
        
        # 现金流量表
        cashflow_df = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
        if cashflow_df is not None and not cashflow_df.empty:
            latest_cashflow = cashflow_df.iloc[0].to_dict()
            result["financial_reports"]["cash_flow"] = [
                {"报表日期": latest_cashflow.get("报告日", ""),
                 "经营活动现金流": latest_cashflow.get("经营活动产生的现金流量净额", "")}
            ]
            result["financial_reports"]["report_date"] = latest_cashflow.get("报告日", "")
            
    except Exception as e:
        print(f"获取财务报告失败: {e}")
    
    return result


def compute_financial_ratios(symbol: str, result: dict) -> dict:
    """
    计算财务比率（PE、PB、ROE等）
    
    Parameters
    ----------
    symbol : str
        股票代码
    result : dict
        结果字典
    
    Returns
    -------
    dict
        更新后的结果字典
    """
    try:
        code = symbol[2:]
        
        # 方式1：通过财务指标接口获取
        try:
            indicator_df = ak.stock_financial_analysis_indicator(symbol=code)
            if indicator_df is not None and not indicator_df.empty:
                latest = indicator_df.iloc[0].to_dict()
                # 可以在financial_reports中添加ratios字段
                result["financial_reports"]["ratios"] = {
                    "roe": latest.get("净资产收益率(%)", None),
                    "eps": latest.get("摊薄每股收益(元)", None),
                    "bps": latest.get("每股净资产(元)", None)
                }
        except:
            pass
            
        # 方式2：通过估值接口获取PE、PB
        try:
            value_df = ak.stock_value_em(symbol=code)
            if value_df is not None and not value_df.empty:
                latest_value = value_df.iloc[-1].to_dict()
                result["financial_reports"]["valuation"] = {
                    "pe": latest_value.get("市盈率", None),
                    "pb": latest_value.get("市净率", None),
                    "ps": latest_value.get("市销率", None)
                }
        except:
            pass
            
    except Exception as e:
        print(f"计算财务比率失败: {e}")
    
    return result


def compute_technical_indicators(result: dict) -> dict:
    """
    计算技术指标（MA、RSI、MACD等）
    
    Parameters
    ----------
    result : dict
        结果字典
    
    Returns
    -------
    dict
        更新后的结果字典
    """
    try:
        daily_data = result["market_data"]["kline_daily"]
        if not daily_data:
            return result
            
        df = pd.DataFrame(daily_data)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # 计算移动平均线（MA）
        if len(close) >= 20:
            # 5日均线
            ma5 = np.convolve(close, np.ones(5)/5, mode='valid')
            indicators['ma5'] = float(ma5[-1]) if len(ma5) > 0 else None
            # 10日均线
            ma10 = np.convolve(close, np.ones(10)/10, mode='valid')
            indicators['ma10'] = float(ma10[-1]) if len(ma10) > 0 else None
            # 20日均线
            ma20 = np.convolve(close, np.ones(20)/20, mode='valid')
            indicators['ma20'] = float(ma20[-1]) if len(ma20) > 0 else None
        
        # 计算RSI（14日）
        if len(close) >= 15:
            rsi = calculate_rsi(close, period=14)
            indicators['rsi14'] = float(rsi[-1]) if len(rsi) > 0 else None
        
        # 计算MACD
        if len(close) >= 26:
            macd, signal, hist = calculate_macd(close)
            indicators['macd'] = {
                'macd': float(macd[-1]) if len(macd) > 0 else None,
                'signal': float(signal[-1]) if len(signal) > 0 else None,
                'histogram': float(hist[-1]) if len(hist) > 0 else None
            }
        
        # 计算布林带（20日）
        if len(close) >= 20:
            upper, middle, lower = calculate_bollinger_bands(close, period=20)
            indicators['bollinger'] = {
                'upper': float(upper[-1]) if len(upper) > 0 else None,
                'middle': float(middle[-1]) if len(middle) > 0 else None,
                'lower': float(lower[-1]) if len(lower) > 0 else None
            }
        
        result["market_data"]["technical_indicators"] = indicators
        
    except Exception as e:
        print(f"计算技术指标失败: {e}")
    
    return result


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    计算RSI指标
    
    Parameters
    ----------
    prices : np.ndarray
        价格序列
    period : int
        计算周期
    
    Returns
    -------
    np.ndarray
        RSI值序列
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = np.nan
    rsi[period] = 100 - 100 / (1 + rs)
    
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100 - 100 / (1 + rs)
    
    return rsi


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    计算MACD指标
    
    Parameters
    ----------
    prices : np.ndarray
        价格序列
    fast : int
        快线周期
    slow : int
        慢线周期
    signal : int
        信号线周期
    
    Returns
    -------
    tuple
        (MACD线, 信号线, 柱状图)
    """
    # 计算EMA
    def ema(data, period):
        alpha = 2 / (period + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = ema(macd, signal)
    histogram = macd - signal_line
    
    return macd, signal_line, histogram


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std: int = 2) -> tuple:
    """
    计算布林带
    
    Parameters
    ----------
    prices : np.ndarray
        价格序列
    period : int
        计算周期
    std : int
        标准差倍数
    
    Returns
    -------
    tuple
        (上轨, 中轨, 下轨)
    """
    middle = np.zeros_like(prices)
    upper = np.zeros_like(prices)
    lower = np.zeros_like(prices)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std_val = np.std(window)
        upper[i] = middle[i] + std * std_val
        lower[i] = middle[i] - std * std_val
    
    # 前period-1个值为NaN
    middle[:period-1] = np.nan
    upper[:period-1] = np.nan
    lower[:period-1] = np.nan
    
    return upper, middle, lower


def get_industry_avg(result: dict) -> dict:
    """
    获取行业平均估值数据
    
    Parameters
    ----------
    result : dict
        结果字典
    
    Returns
    -------
    dict
        更新后的结果字典
    """
    try:
        # 通过申万行业指数获取行业平均估值
        industry_df = ak.sw_index_third_info()
        if industry_df is not None and not industry_df.empty:
            # 计算各指标的平均值作为参考
            avg_pe = industry_df['TTM(滚动)市盈率'].mean()
            avg_pb = industry_df['市净率'].mean()
            
            result["industry_avg"]["pe"] = float(avg_pe) if not pd.isna(avg_pe) else None
            result["industry_avg"]["pb"] = float(avg_pb) if not pd.isna(avg_pb) else None
            
            # ROE可以通过其他方式获取，这里设为示例值
            result["industry_avg"]["roe"] = None
            
    except Exception as e:
        print(f"获取行业平均估值失败: {e}")
    
    return result


def get_index_data(result: dict) -> dict:
    """
    获取主要指数数据
    
    Parameters
    ----------
    result : dict
        结果字典
    
    Returns
    -------
    dict
        更新后的结果字典
    """
    try:
        # 获取所有指数实时行情
        index_spot_df = ak.stock_zh_a_spot_em()
        if index_spot_df is not None and not index_spot_df.empty:
            # 上证指数
            sh_row = index_spot_df[index_spot_df['代码'] == 'sh000001']
            if not sh_row.empty:
                result["index_data"]["sh000001"] = {
                    "price": float(sh_row.iloc[0]['最新价']),
                    "change_pct": float(sh_row.iloc[0]['涨跌幅'])
                }
            
            # 深证成指
            sz_row = index_spot_df[index_spot_df['代码'] == 'sz399001']
            if not sz_row.empty:
                result["index_data"]["sz399001"] = {
                    "price": float(sz_row.iloc[0]['最新价']),
                    "change_pct": float(sz_row.iloc[0]['涨跌幅'])
                }
            
            # 创业板指
            cy_row = index_spot_df[index_spot_df['代码'] == 'sz399006']
            if not cy_row.empty:
                result["index_data"]["sz399006"] = {
                    "price": float(cy_row.iloc[0]['最新价']),
                    "change_pct": float(cy_row.iloc[0]['涨跌幅'])
                }
                
    except Exception as e:
        print(f"获取指数数据失败: {e}")
    
    return result


def format_json_output(result: dict) -> str:
    """
    格式化输出JSON
    
    Parameters
    ----------
    result : dict
        结果字典
    
    Returns
    -------
    str
        格式化的JSON字符串
    """
    return json.dumps(result, ensure_ascii=False, indent=4)


# 使用示例
if __name__ == "__main__":
    
    # 获取贵州茅台的数据
    # data = get_all_data("sh600519")
    # data = get_stock_kline_data("sh600519",result)
    # data = get_realtime_with_name("sh600519",result)
    data = get_all_data("sh600519")
    print(format_json_output(data))