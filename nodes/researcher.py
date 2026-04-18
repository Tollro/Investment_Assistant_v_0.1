"""
Researcher节点：获取名称或代码对应的所有信息，获取的信息自动以json格式保存至全局InvestmentState之中
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
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_openai import AzureChatOpenAI
import os
import time

from typing import Literal
from pydantic import BaseModel, Field

from akshare_tools.Data_Fetch import get_all_data, query_by_name_keyword as _query_by_name_keyword, query_by_code as _query_by_code


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
    fetch_times: int

    # 同步全局 InvestmentState
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    stock_code: Optional[str]
    stock_name: Optional[str]
    collected_data: Optional[Dict[str, Any]]
    data_available: bool


class GetStockAllDataInput(BaseModel):
    symbol: str = Field(description="股票代码（必须包含前缀），如 'sz000001'")
    start_date: str = Field(description="开始日期 YYYYMMDD", default="20250601")
    end_date: str = Field(description="结束日期 YYYYMMDD", default="20260101")


@tool(args_schema=GetStockAllDataInput)
def fetch_data(symbol: str, start_date="20250601", end_date="20260101")->dict:
    """获取指定股票在日期范围内的行情、财务等综合信息，返回JSON格式。"""
    result = get_all_data(symbol, start_date, end_date)
    if result == None:
        return {"error": "数据获取失败，请检查股票代码是否正确或稍后重试"}
    return result


@tool
def get_by_stock_keyword(keyword:str) -> dict:
    """根据股票名称或关键词查找可能的股票代码与名称。若返回多个结果，请让用户选择。"""
    rows = _query_by_name_keyword(keyword=keyword)
    if not rows:
        return {"error": f"未找到与'{keyword}'相关的股票"}
    if len(rows) == 1:
        return {"stock_code": rows[0][0], "stock_name": rows[0][1]}
    # 返回多个选项供 LLM 处理
    return {"multiple_matches": [{"code": r[0], "name": r[1]} for r in rows]}


@tool
def get_by_stock_code(code:str) -> dict:
    """根据股票代码（可含前缀或不含）查找股票名称及标准代码。"""
    rows = _query_by_code(code=code)
    if not rows:
        return {"error": f"未找到代码'{code}'对应的股票"}
    if len(rows) == 1:
        return {"stock_code": rows[0][0], "stock_name": rows[0][1]}
    return {"multiple_matches": [{"code": r[0], "name": r[1]} for r in rows]}


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
    
    # 注意！！需要重新写prompt
    system_prompt="""
你是一名拥有二十年从业经验的专业的股票研究员，负责搜集用户指定股票的市场数据。

**工作流程**：
1. 分析用户输入，提取股票关键词或代码。
2. 若只有名称/关键词，先调用 `get_by_stock_keyword`或'get_by_stock_code' 获取候选股票。
   - 若返回多个匹配，**必须暂停并向用户询问具体选择**（输出问题等待用户回复）。
3. 获取到候选的股票代码与名称后调用 `fetch_data` 获取数据。
   - 若用户未指定时间范围，使用默认日期（20250601-20260101）。
4. 若用户输入模糊或信息不足，主动向用户澄清。

**重要原则**：
- 一次只进行一个操作，避免不必要的工具重复调用。
- 工具返回错误时，尝试修正参数或请求用户协助，不要陷入无限重试。
- 完成数据采集后，无需进一步分析，只需告知“数据已准备完毕”。
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
    return {"messages": [response]}


def update_state_from_tool(state: ResearcherState) -> dict:
    """从工具调用结果中提取数据，更新 collected_data 及相关字段。"""
    messages = state["messages"]
    if not messages:
        return {}

    last_msg = messages[-1]

    # 处理 fetch_data 工具返回
    if isinstance(last_msg, ToolMessage) and last_msg.name == "fetch_data":
        fetch_times = state.get("fetch_times", 0) + 1
        updates = {
            "fetch_times": fetch_times,
            "data_available": False,
            "collected_data": None
        }
        try:
            data = json.loads(last_msg.content)
        except Exception as e:
            print(f"[Researcher] JSON解析失败: {e}")
            updates["collected_data"] = {"raw_response": last_msg.content, "error": str(e)}
            return updates

        # 灵活映射：无论返回结构如何，尽量保存所有内容
        updates["collected_data"] = data
        if "error" not in data:
            updates["data_available"] = True

        # 同时尝试提取股票代码和名称（若 data 中包含）
        if "stock_code" in data:
            updates["stock_code"] = data["stock_code"]
        if "stock_name" in data:
            updates["stock_name"] = data["stock_name"]

        print("[Researcher] 数据采集完成，已更新状态。")
        return updates

    # 处理股票查询工具，自动填充代码与名称
    elif isinstance(last_msg, ToolMessage) and last_msg.name in ("get_by_stock_keyword", "get_by_stock_code"):
        try:
            result = json.loads(last_msg.content)
        except:
            return {}
        if "stock_code" in result and "stock_name" in result:
            return {
                "stock_code": result["stock_code"],
                "stock_name": result["stock_name"]
            }
    return {}


def should_continue(state: ResearcherState) -> Literal["tool_node", "__end__"]:
    """决定是否继续调用工具。"""
    messages = state["messages"]
    last_message = messages[-1]

    # 若LLM要求调用工具，且未超过最大次数
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        if state.get("fetch_times", 0) < 3:
            return "tool_node"
        else:
            print("[Researcher] 已达到最大工具调用次数，强制结束。")
            return END

    # 若最后是工具消息，但未调用fetch_data，可能需要再次询问LLM
    # 这里交由 should_analysis_or_not 处理
    return END


def should_analysis_or_not(state: ResearcherState) -> Literal["llm", "__end__"]:
    """根据意图和当前状态决定下一步。"""
    intent = state.get("intent", "unknown")

    # 如果数据已成功获取，则结束研究员流程
    if state.get("data_available", False):
        print("[Researcher] 数据已就绪，流程结束。")
        return END

    # 对于只查价格的意图，可能只需要简单返回，但当前仍需数据采集
    if intent == "price_check":
        # 如果还没有代码或名称，需要继续询问LLM
        if not state.get("stock_code") and not state.get("stock_name"):
            return "llm"
        return END

    # 其他情况若数据未获取且未超过尝试次数，返回LLM继续
    if state.get("fetch_times", 0) < 3:
        return "llm"
    else:
        print("[Researcher] 数据获取失败且已达最大重试，流程终止。")
        return END


def Researcher_Agent() -> CompiledStateGraph:
    llm = create_llm(temperature=0.2)
    tools = [fetch_data, get_by_stock_keyword, get_by_stock_code]
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools, awrap_tool_call=handle_tool_errors)

    workflow = StateGraph(ResearcherState)

    workflow.add_node("llm", lambda state: call_llm_with_tools(state, llm_with_tools))
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("update_node", update_state_from_tool)

    workflow.add_edge(START, "llm")

    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {"tool_node": "tool_node", END: END}
    )

    workflow.add_edge("tool_node", "update_node")

    workflow.add_conditional_edges(
        "update_node",
        should_analysis_or_not,
        {"llm": "llm", END: END}
    )

    return workflow.compile()


if __name__ == "__main__":
    agent = Researcher_Agent()
    query = "我想查询茅台的相关信息"
    initial_state = {
        "user_query": query,
        "fetch_times": 0,
        "intent": "unknown",
        "stock_code": None,
        "stock_name": None,
        "collected_data": None,
        "data_available": False
    }
    start_time = time.time()
    result = agent.invoke(initial_state)
    end_time = time.time()
    print(f"\n========== 执行完成 ==========")
    print(f"单轮总耗时：{end_time - start_time:.4f} 秒")
    print(f"最终状态：{result}")
    # print_messages_simple(response["result"])