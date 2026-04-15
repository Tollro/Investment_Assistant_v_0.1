"""
ChatBot节点：根据用户的输入判断用户的意图（获取信息/分析股票/提供建议），以及得到用户希望查询的股票名称/代码
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
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
import os

from typing import Literal
from pydantic import BaseModel, Field

from akshare_tools.Data_Fetch import query_by_name_keyword as _query_by_name_keyword, query_by_code as _query_by_code
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

class ChatBotState(TypedDict):
    """ChatBot Agent State"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    # ----- 股票标识 -----
    stock_code: Optional[str]
    stock_name: Optional[str]

@tool
def get_by_stock_keyword(keyword:str) -> Dict[str, str]:
    """根据股票关键词查找股票代码"""
    row = _query_by_name_keyword(keyword=keyword)
    if len(row) == 1:
        return {"stock_code":row[0][0],"stock_name":row[0][1]}
    else:
        return row

@tool
def get_by_stock_code(code:str) -> Dict[str, str]:
    """根据股票代码查找股票名称"""
    row = _query_by_code(code=code)
    if len(row) == 1:
        return {"stock_code":row[0][0],"stock_name":row[0][1]}
    else:
        return row

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

def call_llm_with_tools(state: ChatBotState, llm_with_tools=None) -> dict:
    if llm_with_tools is None:
        raise ValueError("llm_with_tools must be provided to call_llm.")
    
    # 注意！！需要重新写prompt
    system_prompt="""
        你是一名金融咨询客服，你的任务是：
        1. 根据用户的输入判断用户的意图，获得用户想要查询的股票名称或代码，通过查询tools确保用户查询的是一支可查询股票。
        2. 输出 交给后续的调度器执行相应的操作。
        注意：如果调用工具后获得多个结果，则询问用户希望查询哪一只股票；如果查询不到，则回复f"抱歉，目前没有任何关于 {(填写用户输入的关键词)} 的信息。"
    """
    messages = state["messages"]
    # 检查第一条是否为system prompt，若不是则添加
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: ChatBotState) -> Literal["tool_node", "__end__"]:
    """检查最后一条消息，决定下一步去向。"""
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息包含了工具调用，则路由到 "tool_node"
    if last_message.tool_calls:
        return "tool_node"
    # 否则，流程结束
    return END

def Researcher_Agent() -> CompiledStateGraph:

    llm = create_llm(temperature=0.7, max_tokens=1024)
    tools = [get_by_stock_name, get_by_stock_code]
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    workflow = StateGraph(ChatBotState)
    # Use a lambda to pass llm_with_tools to call_llm
    workflow.add_node("llm", lambda state: call_llm_with_tools(ChatBotState, llm_with_tools=llm_with_tools))
    workflow.add_node("tool_node", tool_node)

    workflow.add_edge(START, "llm")
    # workflow.add_edge("tool_node","llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    workflow.add_edge("tool_node", "llm")

    Agent = workflow.compile()
    return Agent


if __name__ == "__main__":

    Agent = Researcher_Agent()

    # ！！注意：输入全局状态中的stock_code，在上层应用中调用invoke
    response = Agent.invoke({"messages": ["获取000012从20250401到20250601的所有信息"]})
    print_messages_simple(response["messages"])

