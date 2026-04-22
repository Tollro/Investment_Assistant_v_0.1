"""
父图文件，连接各子图
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated, Union
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import time
import datetime

from nodes.Researcher import Researcher_Agent
from nodes.Analyst import Analyst_Graph
from nodes.Advisor import Advisor_Graph
from nodes.Responder import Responder_Graph
from nodes.Supervisor import Supervisor_Graph


WorkerType = Literal["Supervisor", "Researcher", "Analyst", "Advisor"]

class InvestmentState(TypedDict):
    # 全局状态
    conversation_history: Annotated[List[BaseMessage], add_messages]
    # messages: Annotated[List[BaseMessage], add_messages]
    history_query: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
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
    response: str                        # Responder 节点输出

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
        

def Update_State(state:InvestmentState) -> str:
    """
    更新全局状态
    """
    current_query  = state.get("user_query","")
    response_text = state.get("response", "")
    stock_name = state.get("stock_name", "")
    analysis_text = state.get("analysis","")
    advices_text = state.get("advices","")

    # 向 history_query 追加一条 HumanMessage（用于记录已处理的问题）
    history_update = [HumanMessage(content=current_query)] if current_query else []
    print(f"\n===关于{stock_name}的分析===\n{analysis_text}")
    print(f"\n===关于{stock_name}的建议===\n{advices_text}")
    print(f"================\n金融客服🤖：\n{response_text}")
    return{
        "history_query": history_update,
        "user_query": "",                     # str，空字符串
        "intent": "unknown",                  # Literal 成员
        "last_worker": None,                  # Optional 可为 None
        "next_worker": None,                  # Optional 可为 None
        "stock_code": None,
        "stock_name": None,
        "collected_data": None,
        "data_available": False,              # bool，设为 False
        "analysis": "",                       # str，空字符串
        "advices": {},                        # Dict，空字典
        "response": "",                       # str，空字符串
    }

def Stock_Graph_Single() -> CompiledStateGraph:
    workflow = StateGraph(InvestmentState)
    
    workflow.add_node("Supervisor", Supervisor_Graph())
    workflow.add_node("Researcher", Researcher_Agent())
    workflow.add_node("Analyst", Analyst_Graph())
    workflow.add_node("Advisor", Advisor_Graph())
    workflow.add_node("Responder", Responder_Graph())
    workflow.add_node("Update_State", Update_State)

    workflow.add_edge(START, "Supervisor")
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Analyst", "Supervisor")
    workflow.add_edge("Advisor", "Supervisor")

    workflow.add_conditional_edges(
        "Supervisor",              
        next_step_judgment
    )

    workflow.add_edge("Responder", "Update_State")
    workflow.add_edge("Update_State", END)
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer = checkpointer)


if __name__ == "__main__":
    print(f"[Researcher] 系统提示中的今日日期: {datetime.date.today().strftime('%Y-%m-%d')}")
    Assistant = Stock_Graph_Single()
    config = {"configurable": {"thread_id": "user_session_123"}}

    print("===== 投资助手（支持多轮对话与中断） =====")
    print("输入 'exit' 退出对话\n")

    state = {
        "user_query": "",
        "conversation_history": []
    }

    def run_stream(input_state, config):
        """
        递归处理 stream 中的中断，返回最终的 final_response 字符串（或 None）。
        """
        for event in Assistant.stream(input_state, config):
            if "__interrupt__" in event:
                interrupt_info = event["__interrupt__"][0]
                data = interrupt_info.value
                print(f"\n[助手]: {data['question']}")
                user_response = input("用户: ")
                # 递归恢复，并直接返回递归结果
                return run_stream(Command(resume=user_response), config)
            else:
                for node_output in event.values():
                    if isinstance(node_output, dict) and "final_response" in node_output:
                        return node_output["final_response"]
        return None

    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == "exit":
            break

        state["user_query"] = user_input
        user_msg = HumanMessage(content=user_input)
        state["conversation_history"] = [user_msg]

        final_response = run_stream(state, config)

        # 获取最新状态（确保下一轮能延续）
        final_state = Assistant.get_state(config).values
        state = final_state

        if final_response:
            print(f"\n[助手]: {final_response}")
        elif final_state.get("advices"):
            # 如果有 advices 但无 final_response，可自行格式化输出
            print(f"\n[助手]: {final_state['advices']}")