from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import re

class InvestmentState(TypedDict):
    """
    结构示例：
    
    "investment_data":
    {
        "stock_code": "sh600519",
        "stock_name": "贵州茅台酒股份有限公司",

        # ----- 行情数据（市场原始数据）-----
        "market_data": {
            "realtime": {
                "price": 1680.00,
                "change_pct": 1.25,
                "volume": 2450000,
                "high": 1695.00,
                "low": 1665.00,
                "timestamp": "2026-04-13 14:30:00"
            },
            "kline_daily": [],    # dataframe转JSON
            "kline_weekly": [],   # dataframe转JSON
            "kline_monthly": []   # dataframe转JSON
        },

        # ----- 财务报表数据（原始文本或半结构化数据）-----
        "financial_reports": {
            "balance_sheet": "原始资产负债表DataFrame转JSON",
            "income_statement": "原始利润表文本DataFrame转JSON",
            "cash_flow": "原始现金流量表文本DataFrame转JSON",
            "report_date": "2025-12-31"
        },

        # # ----- 新闻舆情数据 -----
        # "news_data": {
        #     "titles": [
        #         "贵州茅台一季度净利润增长18%",
        #         "北向资金连续5日净买入贵州茅台"
        #     ],
        #     "key_data": [],   # 存储完整新闻关键数据
        #     "urls": []
        #     "source": "eastmoney"
        # },

        # ----- 其他可选扩展数据 -----
        "industry_avg": {          # 行业均值（用于估值对比）
            "pe": 28.5,
            "pb": 5.2,
            "roe": 0.15
        },
        "index_data": {            # 大盘指数数据（用于β系数计算）
            "sh000001": {"price": 3300.0, "change_pct": 0.5}
        }
    }
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    stock_code: Optional[str]
    stock_name: Optional[str]
    collected_data: Optional[Dict[str, Any]]
    technical_indicators: Optional[Dict[str, Any]]
    fundamental_analysis: Optional[Dict[str, Any]]
    sentiment_summary: Optional[Dict[str, Any]]
    analysis_summary: Optional[str]
    recommendation: Optional[str]
    confidence_score: Optional[float]
    risk_score: Optional[float]
    risk_factors: Optional[List[str]]
    final_decision: Optional[str]
    error_info: Optional[str]
    retry_count: int
    final_response: Optional[str]

# ---------- 辅助校验函数 ----------
def validate_stock_code(code: str) -> bool:
    """校验股票代码格式：交易所后缀（SH/SZ/BJ）+ 6位数字，如 sh600010"""
    pattern = r"^(sh|sz|bj)\d{6}$"
    return bool(re.match(pattern, code))

def validate_collected_data(data: Dict) -> bool:
    """检查 collected_data 是否包含最小必要字段"""
    required_top_keys = {"stock_code", "market_data", "financial_reports", "news_data"}
    if not required_top_keys.issubset(data.keys()):
        return False
    market = data["market_data"]
    if "kline_daily" not in market or len(market["kline_daily"]) < 30:
        return False
    return True