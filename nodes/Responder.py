"""
Responder节点：根据历史消息生成回复
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
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_openai import AzureChatOpenAI
import os
import time
from pydantic import BaseModel, Field


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


class ResearcherState(TypedDict):
    """Researcher Agent State"""
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_history: Annotated[List[BaseMessage], add_messages]
    fetch_times: int

    # 同步全局 InvestmentState
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    stock_code: Optional[str]
    stock_name: Optional[str]
    collected_data: Optional[Dict[str, Any]]
    data_available: bool


def create_llm(temperature: float) -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
                api_key=os.getenv("AZURE_GPT4O_API_KEY"),
                api_version="2025-01-01-preview",
                model="gpt-4o",
                temperature=temperature
            )
    if not llm:
        raise ValueError("Azure OpenAI 模型初始化失败，请检查环境变量设置。")
    else:
        # print("✅ Azure OpenAI 模型初始化成功！")
        return llm


def call_llm_with_tools(state: ResearcherState, llm_with_tools=None) -> dict:
    if llm_with_tools is None:
        raise ValueError("llm_with_tools must be provided to call_llm.")
    
    # 如果股票已选定但数据未就绪，直接生成工具调用，不经过LLM
    stock_code = state.get("stock_code")
    stock_name = state.get("stock_name")
    data_available = state.get("data_available", False)
    
    if stock_code and stock_name and not data_available:
        print(f"[Researcher] 检测到已选定股票 {stock_name}({stock_code})，直接调用 fetch_data 工具。")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "name": "fetch_data",
                "args": {
                    "symbol": stock_code,
                    "start_date": "20250601",
                    "end_date": "20260101"
                },
                "id": f"call_fetch_{int(time.time())}"  # 简单生成唯一ID
            }]
        )
        return {
            "messages": [tool_call_msg],
            "conversation_history": [tool_call_msg]
        }
    
    # 注意！！需要重新写prompt
    system_prompt=f"""
你是一名拥有二十年从业经验的专业的股票研究员，负责搜集用户指定股票的市场数据。你收集到的数据会交给专业的分析师进行分析。
用户意图：{state.get("intent", "unkonwn")}

**工作流程**：
1. 分析用户输入，提取股票关键词或代码。若只有名称/关键词，先调用 `get_by_stock_keyword`或'get_by_stock_code' 获取候选股票。
   - 如果工具返回多个匹配，系统会自动向用户询问选择，你无需处理。
2. 获取到候选的股票代码与名称后调用 `fetch_data` 获取数据。
   - 若用户未指定时间范围，使用默认日期（20250601-20260101）。

**重要原则**：
- 若用户输入模糊或信息不足，主动向用户澄清。
- 一次只进行一个操作，避免不必要的工具重复调用。
- 工具返回错误时，尝试修正参数或请求用户协助，不要陷入无限重试。
- 完成数据采集后，无需进一步分析，只需告知“XX(股票名称)的数据已准备完毕”。
"""

    messages = state.get("messages", [])
    if not messages:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_query"])
        ]
    else:
        if not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "conversation_history": [response]
    }


def Researcher_Agent() -> CompiledStateGraph:
    llm = create_llm(temperature=0.2)

    llm_with_tools = llm.bind_tools(tools)

    workflow = StateGraph(ResearcherState)

    workflow.add_node("llm", lambda state: call_llm_with_tools(state, llm_with_tools))
    workflow.add_node("tool_node", tool_node)


    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    agent = Researcher_Agent()
    query = "我想查询华的相关信息"
    initial_state = {
        "user_query": query,
        "fetch_times": 0,
        "intent": "price_check",
        "stock_code": None,
        "stock_name": None,
        "collected_data": None,
        "data_available": False,
        "conversation_history": []
    }
    start_time = time.time()

    # 使用 stream 处理可能的中断
    config = {"configurable": {"thread_id": "test_researcher"}}

    def run_stream(input_state, config):
        for event in agent.stream(input_state, config):
            if "__interrupt__" in event:
                interrupt_obj = event["__interrupt__"][0]
                data = interrupt_obj.value
                print("\n[中断] 问题：", data["question"])
                user_response = input("你的回答：")
                # 用 Command 恢复执行，继续递归处理可能的新中断
                run_stream(Command(resume=user_response), config)
                return  # 恢复后本次 stream 结束，由递归调用的 stream 接管后续事件
            else:
                # 打印非中断事件，便于观察流程
                print(event)
    
    run_stream(initial_state, config)
    end_time = time.time()
    
    print(f"\n========== 执行完成 ==========")
    print(f"总耗时：{end_time - start_time:.4f} 秒")
    # print(f"最终状态：{result["messages"][-1].content}")
    # print_messages_simple(result["messages"])