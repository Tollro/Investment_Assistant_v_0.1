"""
父图文件，连接各子图
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph.message import add_messages ,StateGraph
from langchain_core.messages import BaseMessage
from nodes.Researcher import Researcher_Agent
from nodes.Supervisor import supervisor_Graph

class InvestmentState(TypedDict):
    # 全局状态
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
    # ----- Supervisor 流程控制字段 -----
    needs_clarification: bool          # 是否需要暂停并向用户追问
    clarification_question: str        # 向用户展示的追问内容
    current_phase: Literal["collecting", "analyzing", "reporting", "interrupted"]
    last_worker: Optional[str]         # 记录最后执行的子图名称，用于断点恢复

    # ----- 股票标识 -----
    stock_code: Optional[str]
    stock_name: Optional[str]

    # ----- 原始采集数据（结构完全匹配示例）-----
    collected_data: Optional[Dict[str, Any]]  # 内部结构详见注释

    # ----- 中间分析结果 -----
    technical_indicators: Optional[Dict[str, Any]]
    fundamental_analysis: Optional[Dict[str, Any]]
    sentiment_summary: Optional[Dict[str, Any]]

    # ----- 最终输出内容 -----
    analysis_summary: Optional[str]
    recommendation: Optional[str]
    confidence_score: Optional[float]
    risk_score: Optional[float]
    risk_factors: Optional[List[str]]
    final_decision: Optional[str]

    # ----- 流程控制 -----
    error_info: Optional[str]
    retry_count: int
    final_response: Optional[str]
    """
    collected_data 字段内部结构说明（类型为 Dict[str, Any] 时的参考）：
    {
        "market_data": {
            "realtime": {
                "price": 1680.00,
                "change_pct": 1.25,
                "volume": 2450000,
                "high": 1695.00,
                "low": 1665.00,
                "timestamp": "2026-04-13 14:30:00"
            },
            "kline_daily": [],     # List[Dict]
            "kline_weekly": [],    # List[Dict]
            "kline_monthly": []    # List[Dict]
        },
        "financial_reports": {
            "balance_sheet": [],      # List[Dict]
            "income_statement": [],   # List[Dict]
            "cash_flow": [],          # List[Dict]
            "report_date": "2025-12-31"
        },
        # "news_data": {
        #     "titles": [],
        #     "key_data": [],
        #     "urls": [],
        #     "source": "eastmoney"
        # },
        "industry_avg": {
            "pe": 28.5,
            "pb": 5.2,
            "roe": 0.15
        },
        "index_data": {
            "sh000001": {"price": 3300.0, "change_pct": 0.5}
        }
    }
    """

# # 创建父图、子图
# def build_parent_graph():
#     workflow = StateGraph(InvestmentState)

#     researcher_graph = build_researcher_graph()
#     workflow.add_node("researcher", researcher_graph)

#     # ...其他节点和边
#     return workflow.compile()

    