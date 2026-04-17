"""
Analyst节点：获取名称或代码对应的所有信息，获取的信息自动以json格式保存至全局InvestmentState之中
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
import os

from typing import Literal
from pydantic import BaseModel, Field

from nodes.graph import InvestmentState

def print_messages_simple(messages):
    """简洁打印消息列表，只显示类型、内容和工具调用信息"""
    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        print(f"\n--- 第 {i+1} 条消息 ({msg_type}) ---")
        
        if msg_type == "HumanMessage":
            print(f"内容: {msg.content}")
        elif msg_type == "AIMessage":
            if msg.content:
                print(f"内容: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"调用工具: {tc['name']}, 参数: {tc['args']}")
        elif msg_type == "ToolMessage":
            print(f"工具名称: {msg.name}")

            # 如果返回内容过长可截断显示
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
                
            print(f"返回内容: {content}")
        else:
            print(f"agent回复内容: {msg.content}")

class AnalystState(TypedDict):
    """Analyst Agent State"""
    messages: Annotated[List[BaseMessage], add_messages]

    # 同步全局 InvestmentState
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    stock_code: Optional[str]
    stock_name: Optional[str]
    collected_data: Optional[Dict[str, Any]]
    data_available: bool

class GetStockAllDataInput(BaseModel):
    symbol: str = Field(description="股票代码（必须包含前缀），如 'sz000001'")
    start_date: str = Field(description="开始日期，格式 YYYYMMDD（默认20250601）")
    end_date: str = Field(description="结束日期，格式 YYYYMMDD（默认20260101）")

# @tool(args_schema=GetStockAllDataInput)
# def fetch_data(symbol: str, start_date="20250601", end_date="20260101")->dict:
#     """获取指定股票代码在指定日期范围内的数据，包括行情、财务等综合信息，输出JSON格式。"""
#     result = get_all_data(symbol, start_date, end_date)
#     if result == None:
#         print("\nfetch_data工具调用结果为空")
#     return result

# @tool
# def get_by_stock_keyword(keyword:str) -> list[str]:
#     """根据股票名称查找股票代码"""
#     row = _query_by_name_keyword(keyword=keyword)
#     if len(row) == 1:
#         return {"stock_code":row[0][0],"stock_name":row[0][1]}
#     else:
#         return row

# @tool
# def get_by_stock_code(code:str) -> list[str]:
#     """根据股票代码查找股票名称"""
#     row = _query_by_code(code=code)
#     if len(row) == 1:
#         return {"stock_code":row[0][0],"stock_name":row[0][1]}
#     else:
#         return row

@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def create_llm(temperature: float, max_tokens: int) -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
                api_key=os.getenv("AZURE_GPT4O_API_KEY"),
                api_version="2025-01-01-preview",
                model="gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens
            )
    if not llm:
        raise ValueError("Azure OpenAI 模型初始化失败，请检查环境变量设置。")
    else:
        # print("✅ Azure OpenAI 模型初始化成功！")
        return llm

def call_llm_analysis(state: AnalystState, llm=None) -> dict:
    if llm is None:
        raise ValueError("llm must be provided to call_llm_analysis.")
    
    # 注意！！需要重新写prompt
    system_prompt="""
    你是一名拥有二十年从业经验的专业的股票分析员。
    你的任务是：根据获取到的股票数据以及各项指标对股票（或企业）的发展前景、风险等等信息进行分析。

    注意：
        1. 一定要以中立、客观的角度分析，不得带有个人主观看法。
    """

    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    stock_name = state.get("stock_name", "")
    stock_code = state.get("stock_code", "")
    collected_data = state.get("collected_data", "")

    # 如果 messages 为空，说明是首次调用，需要构造初始对话
    if not messages:
        human_message = f"""
            用户输入：{user_query}
            当前已获得信息如下：
                股票名称：{stock_name}
                股票代码：{stock_code}
                股票数据：{collected_data}
            请对该股票进行分析。
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]
    else:
        # 已有历史消息（如工具调用后的循环），确保系统提示在最前面
        if not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

    response = llm.invoke(messages)
    return {"messages": [response]}

def update_state_from_tool(state: AnalystState) -> dict:
    """
    当最后一条消息是 fetch_data 工具返回的 ToolMessage 时，
    将其内容解析为 JSON 并更新到全局状态的对应字段中。
    """
    # print("\n⚙️ 进入 update_state_from_tool 节点")  # 调试用
    messages = state["messages"]
    if not messages:
        return {}
    last_msg = messages[-1]

    # ############## 调试用 ##############
    # print(f"最后一条消息类型: {type(last_msg).__name__}")
    # if isinstance(last_msg, ToolMessage):
    #     print(f"调用的工具名称: {last_msg.name}")
    # ############## 调试用 ##############

    
    # 同名字段会自动更新到父节点
    if isinstance(last_msg, ToolMessage) and last_msg.name == "fetch_data":
        fetch_times = state["fetch_times"] + 1
        updates = {
            "fetch_times": fetch_times,
            "collected_data":{}
        }
        try:
            data = json.loads(last_msg.content)
        except Exception:
            # 如果解析失败，保持原样
            return {}
        
        # 根据 get_all_data 实际返回的字段名进行映射
        if "market_data" in data:
            updates["collected_data"] = data
        # if "financial_reports" in data:
        #     updates["collected_data"]["financial_reports"] = data["financial_reports"]
        # if "industry_avg" in data:
        #     updates["collected_data"]["industry_avg"] = data["industry_avg"]
        # if "index_data" in data:
        #     updates["collected_data"]["index_data"] = data["index_data"]
        # if "stock_code" in data:
        #     updates["stock_code"] = data["stock_code"]
        # if "stock_name" in data:
        #     updates["stock_name"] = data["stock_name"]
        # print("-"*20,"\n",updates)
        print("\n----已更新 Reseaercher 状态----")
        updates["data_available"] = True
        return updates
    return {}

def should_continue(state: AnalystState) -> Literal["tool_node", "__end__"]:
    """检查最后一条消息，决定下一步去向。"""
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息包含了工具调用，则路由到 "tool_node"
    if last_message.tool_calls and state["fetch_times"] <= 3:
        return "tool_node"
    # 否则，流程结束
    return END

# 判断下一个节点应该往哪里走
def should_analysis_or_not(state: AnalystState) -> Literal["__end__", "llm"]:
    intent = state["intent"]
    if intent == "price_check":
        return "llm"
    else:
        if state["data_available"]:
            print("\n已成功获取数据，进入分析环节......")
            return END
        else:
            return "llm"

def Analyst_Agent() -> CompiledStateGraph:

    llm = create_llm(temperature=0.2, max_tokens=1024)

    workflow = StateGraph(AnalystState)
    # Use a lambda to pass llm_with_tools to call_llm
    workflow.add_node("llm", lambda state: call_llm_analysis(state, llm))
    # workflow.add_node("tool_node", tool_node)
    workflow.add_node("update_node", update_state_from_tool)

    workflow.add_edge(START, "llm")
    # workflow.add_edge("tool_node","llm")
    # workflow.add_conditional_edges(
    #     "llm",
    #     should_continue,
    #     {
    #         "tool_node": "tool_node",
    #         END: END
    #     }
    # )
    # workflow.add_edge("tool_node", "update_node")
    # workflow.add_conditional_edges(
    #     "update_node",
    #     should_analysis_or_not,
    #     {
    #         "llm": "llm",
    #         END: END
    #     }
    # )
    Agent = workflow.compile()
    return Agent


if __name__ == "__main__":

    Agent = Analyst_Agent()

    query = "我想查询茅台的相关信息"
    initial_state = {
        "user_query": query,
        "fetch_times": 0,
        "intent": "unknown",
        "stock_code": None,
        "stock_name": None,
        "collected_data": None,
        "data_available": False
        # "messages" 字段可以不传，让 call_llm 内部构建
    }
    response = Agent.invoke(initial_state)
    print_messages_simple(response["messages"])