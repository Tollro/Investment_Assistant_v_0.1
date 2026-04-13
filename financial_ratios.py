import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 第一部分：财务指标计算函数
# ------------------------------
def compute_financial_ratios(df_fin):
    """
    输入：季度/年度财务报表DataFrame，必须包含以下列：
    date, revenue, cost, net_profit, total_assets, total_liabilities,
    current_assets, current_liabilities, inventory, receivables, equity
    """
    df = df_fin.sort_values('date').copy()
    
    # 计算同比所需的去年同期数据（通过shift）
    df['revenue_prev_year'] = df['revenue'].shift(4)  # 假设季度数据，4个季度为一年
    df['net_profit_prev_year'] = df['net_profit'].shift(4)
    
    # 平均值的计算（期初+期末平均，这里用本季度与上季度平均，更精确可用期初期末）
    df['avg_equity'] = (df['equity'] + df['equity'].shift(1)) / 2
    df['avg_assets'] = (df['total_assets'] + df['total_assets'].shift(1)) / 2
    df['avg_inventory'] = (df['inventory'] + df['inventory'].shift(1)) / 2
    df['avg_receivables'] = (df['receivables'] + df['receivables'].shift(1)) / 2
    
    # 计算各项比率
    df['gross_margin'] = (df['revenue'] - df['cost']) / df['revenue']
    df['net_margin'] = df['net_profit'] / df['revenue']
    df['roe'] = df['net_profit'] / df['avg_equity']
    df['roa'] = df['net_profit'] / df['avg_assets']
    df['current_ratio'] = df['current_assets'] / df['current_liabilities']
    df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
    df['debt_to_asset'] = df['total_liabilities'] / df['total_assets']
    df['inventory_turnover'] = df['cost'] / df['avg_inventory']   # 年化需乘以4（季度数据）
    df['receivable_turnover'] = df['revenue'] / df['avg_receivables']
    df['revenue_yoy'] = (df['revenue'] - df['revenue_prev_year']) / df['revenue_prev_year']
    df['profit_yoy'] = (df['net_profit'] - df['net_profit_prev_year']) / df['net_profit_prev_year']
    
    # 保留需要的列
    ratio_cols = ['date', 'gross_margin', 'net_margin', 'roe', 'roa', 'current_ratio',
                  'quick_ratio', 'debt_to_asset', 'inventory_turnover', 'receivable_turnover',
                  'revenue_yoy', 'profit_yoy']
    return df[ratio_cols]

# ------------------------------
# 第二部分：技术指标计算函数
# ------------------------------
def compute_technical_indicators(df_kline):
    """输入：日K线DataFrame，包含 date, open, high, low, close, volume"""
    df = df_kline.sort_values('date').copy()
    
    # 移动平均
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # EMA
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    # RSI (14日)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # 布林带 (20日)
    df['BB_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std
    
    # ATR (14日)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    
    # OBV (能量潮)
    df['price_change'] = df['close'].diff()
    df['obv_direction'] = np.where(df['price_change'] > 0, 1,
                                   np.where(df['price_change'] < 0, -1, 0))
    df['OBV'] = (df['obv_direction'] * df['volume']).cumsum()
    
    # 成交量均线
    df['volume_MA5'] = df['volume'].rolling(5).mean()
    
    # 删除辅助列
    df.drop(['price_change', 'obv_direction'], axis=1, inplace=True)
    return df

# ------------------------------
# 第三部分：估值指标计算（合并财务与股价）
# ------------------------------
def compute_valuation_ratios(df_fin, df_kline):
    """
    将季度财务数据与日K线结合计算PE/PB/PS。
    财务数据按季度日期向前填充至每日。
    需要财务表包含：date, net_profit, equity, revenue, total_shares (总股本)
    需要K线表包含：date, close
    """
    # 准备财务每日填充
    df_fin = df_fin.sort_values('date').copy()
    df_kline = df_kline.sort_values('date').copy()
    
    # 合并：将财务日期与最近的K线日期对齐（向前填充）
    merged = pd.merge_asof(df_kline[['date', 'close']],
                           df_fin[['date', 'net_profit', 'equity', 'revenue', 'total_shares']],
                           on='date', direction='backward')
    
    # 计算每股指标（假设财务数据为单季度累计值，需年化EPS）
    # 注意：此处简化处理，通常EPS为TTM（最近四个季度累计），这里用最近一季*4近似
    merged['EPS_ttm'] = merged['net_profit'] * 4 / merged['total_shares']
    merged['BVPS'] = merged['equity'] / merged['total_shares']
    merged['SPS_ttm'] = merged['revenue'] * 4 / merged['total_shares']
    
    merged['PE'] = merged['close'] / merged['EPS_ttm']
    merged['PB'] = merged['close'] / merged['BVPS']
    merged['PS'] = merged['close'] / merged['SPS_ttm']
    
    # 只保留日期和估值指标
    return merged[['date', 'PE', 'PB', 'PS']]

# ------------------------------
# 主程序：读取数据、计算、导出
# ------------------------------
if __name__ == "__main__":
    # 请根据实际文件路径修改
    FIN_PATH = 'financials.csv'
    KLINE_PATH = 'kline_daily.csv'
    OUTPUT_PATH = 'financial_analysis_output.xlsx'
    
    # 读取数据
    df_fin_raw = pd.read_csv(FIN_PATH, parse_dates=['date'])
    df_kline_raw = pd.read_csv(KLINE_PATH, parse_dates=['date'])
    
    # 计算财务指标
    df_fin_ratios = compute_financial_ratios(df_fin_raw)
    
    # 计算技术指标
    df_kline_tech = compute_technical_indicators(df_kline_raw)
    
    # 计算估值指标（注意财务表需要 total_shares 字段，若没有需自行补充）
    # 若没有股本数据，可先跳过估值部分
    if 'total_shares' in df_fin_raw.columns:
        df_valuation = compute_valuation_ratios(df_fin_raw, df_kline_raw)
        # 将估值指标合并回K线表（按日期对齐）
        df_kline_final = pd.merge(df_kline_tech, df_valuation, on='date', how='left')
    else:
        df_kline_final = df_kline_tech
        print("警告：财务数据缺少 'total_shares' 字段，跳过估值指标计算。")
    
    # 导出为Excel多Sheet
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        df_fin_ratios.to_excel(writer, sheet_name='财务比率', index=False)
        df_kline_final.to_excel(writer, sheet_name='K线技术指标', index=False)
        # 可额外保存原始数据副本
        df_fin_raw.to_excel(writer, sheet_name='原始财务数据', index=False)
        df_kline_raw.to_excel(writer, sheet_name='原始K线数据', index=False)
    
    print(f"分析完成！结果已保存至 {OUTPUT_PATH}")
    print("财务比率最新一期数据：")
    print(df_fin_ratios.tail(1).to_string())