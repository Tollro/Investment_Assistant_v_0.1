"""
父图文件，连接各子图
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_openai import AzureChatOpenAI
import os
import time

from typing import Literal, Union

from nodes.Researcher import Researcher_Agent
from nodes.Analyst import Analyst_Graph
from nodes.Advisor import Advisor_Graph
from nodes.Supervisor import Supervisor_Graph


WorkerType = Literal["Supervisor", "Researcher", "Analyst", "Advisor"]

class InvestmentState(TypedDict):
    # 全局状态
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
    # ----- Supervisor 流程控制字段 -----
    needs_clarification: bool
    clarification_question: str
    current_phase: Literal["collecting", "analyzing", "reporting", "interrupted"]
    
    last_worker: Optional[WorkerType]
    next_worker: Optional[Union[WorkerType, Literal["__end__"]]]          # 可以是子图名或 "end"

    # ----- 股票标识 -----
    stock_code: Optional[str]
    stock_name: Optional[str]

    # ----- 原始采集数据 -----
    collected_data: Optional[Dict[str, Any]]
    data_available: bool                 # Researcher 节点会设置

    # ----- 中间分析结果 -----
    analysis: str                        # Analyst 节点输出
    advices: Dict[str, Any]              # Advisor 节点输出

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


def next_step_judgment(state: InvestmentState) -> str:
    """
    根据 Supervisor 设置的 next_worker 决定路由目标。
    """
    next = state.get("next_worker")
    if next:
        return next
    else:
        print("==== 错误 ====\n找不到next_worker!")
        # return {}

def Stock_Graph_Single() -> CompiledStateGraph:
    workflow = StateGraph(InvestmentState)
    
    workflow.add_node("Supervisor", Supervisor_Graph())
    workflow.add_node("Researcher", Researcher_Agent())
    workflow.add_node("Analyst", Analyst_Graph())
    workflow.add_node("Advisor", Advisor_Graph())

    workflow.add_edge(START, "Supervisor")
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Analyst", "Supervisor")
    workflow.add_edge("Advisor", "Supervisor")

    workflow.add_conditional_edges(
        "Supervisor",              
        next_step_judgment
    )

    return workflow.compile()


if __name__ == "__main__":
    Assistant = Stock_Graph_Single()

    query = "我想知道茅台走势"
    initial_state={
        "user_query": query,
    }

    start_time = time.time()
    result = Assistant.invoke(initial_state)
    end_time = time.time()
    
    print(f"\n========== 执行完成 ==========")
    print(f"单轮总耗时：{end_time - start_time:.4f} 秒")
    print(f"最终状态：{result}")