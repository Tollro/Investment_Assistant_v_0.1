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
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain_openai import AzureChatOpenAI
import os
import time
from pydantic import BaseModel, Field
import datetime

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
    conversation_history: Annotated[List[BaseMessage], add_messages]
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
    start_date: str = Field(description="开始日期 YYYYMMDD")
    end_date: str = Field(description="结束日期 YYYYMMDD")


@tool(args_schema=GetStockAllDataInput)
def fetch_data(symbol: str, start_date=(datetime.date.today()-datetime.timedelta(days=182)).strftime("%Y%m%d"), end_date=datetime.date.today().strftime("%Y%m%d"))->dict:
    """获取指定股票在日期范围内的行情、财务等综合信息，返回JSON格式。"""
    print(f"[fetch_data] 调用参数: symbol={symbol}, start={start_date}, end={end_date}")
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
                temperature=temperature,
                timeout=15,           # 请求超时（秒）
                max_retries=4,   # 重试次数
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
    
    # 注意！！需要重新写prompt
    system_prompt=f"""
你是一名拥有二十年从业经验的专业的股票研究员，负责搜集用户指定股票的市场数据。你收集到的数据会交给专业的分析师进行分析。
**重要日期说明**：
- 今日日期是 {datetime.date.today().strftime("%Y-%m-%d")},不是2024年!
用户提问：{state.get("user_query", "")}

**当前状态**：
- 已确定股票：{stock_name} ({stock_code}) 
- 数据是否已采集：{data_available}

**工作流程**：
1. 如果股票代码或名称未确定，先调用 `get_by_stock_keyword` 或 `get_by_stock_code` 确定股票。
2. 一旦股票确定，**必须立即调用 `fetch_data` 工具**来获取数据。`fetch_data` 工具的参数：
   - `symbol`：必填，股票代码。
   - `start_date` 和 `end_date`：选填，若用户未指定，**不要传递这两个参数**，工具会自动使用最近两个月的日期。
3. 获取到数据后，无需分析，只需回复“数据已准备完毕”。

**重要原则**：
- 即使股票已经选定，你也必须主动调用 `fetch_data`，不能只输出文字。
- 一次只进行一个工具调用，避免重复。
- 工具返回错误时，尝试修正参数重试一次，若仍失败则告知用户。
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


def update_state_from_tool(state: ResearcherState) -> dict:
    """从工具调用结果中提取数据，更新 collected_data 及相关字段。"""
    messages = state["messages"]
    if not messages:
        return {}

    last_msg = messages[-1]
    updates = {}

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

        # 增强校验：若明确有 error 字段，或关键价格字段缺失，则标记为不可用
        if "error" in data:
            updates["data_available"] = False
        else:
            # 检查实时价格是否有效（可根据实际数据结构调整）
            realtime = data.get("market_data", {}).get("realtime", {})
            price = realtime.get("price")
            # 不要求一定有日线数据
            if price is None or price == 0:
                print("[Researcher] 警告：未获取到有效价格数据")
                updates["data_available"] = True
            else:
                updates["data_available"] = True

        # 同时尝试提取股票代码和名称（若 data 中包含）
        if "stock_code" in data:
            updates["stock_code"] = data["stock_code"]
        if "stock_name" in data:
            updates["stock_name"] = data["stock_name"]

        if updates["data_available"]:
            print("[Researcher] 数据采集成功，已更新状态。")
        else:
            print("[Researcher] 数据采集失败，将尝试重试。")
        return updates

    # 处理股票查询工具，自动填充代码与名称
    elif isinstance(last_msg, ToolMessage) and last_msg.name in ("get_by_stock_keyword", "get_by_stock_code"):
        try:
            result = json.loads(last_msg.content)
        except:
            return {}
        # 情况1：错误信息 - 未找到股票
        if "error" in result and "未找到" in result["error"]:
            error_msg = result["error"]
            assistant_prompt = AIMessage(content=f"{error_msg} 请提供更准确的股票名称或代码。")
            # 触发中断，等待用户重新输入
            user_new_input = interrupt({
                "type": "not_found",
                "question": f"{error_msg} 请重新输入股票名称或代码：",
                "original_query": state.get("user_query", "")
            })
            # 用户输入后，将用户的新输入作为 HumanMessage 加入历史和消息流
            user_msg = HumanMessage(content=user_new_input)
            return {
                "conversation_history": [assistant_prompt, user_msg],
                "messages": [user_msg],          # 让LLM继续处理新输入
                "user_query": user_new_input,    # 更新当前查询
                "fetch_times": 0,                # 重置重试计数
            }
        elif "multiple_matches" in result:
            options = result["multiple_matches"]
            # 构建选项列表文本（仅用于展示，不存入历史）
            option_lines = [f"{i+1}. {opt['name']} ({opt['code']})" for i, opt in enumerate(options)]
            question_content = "找到多个匹配的股票，请选择：\n" + "\n".join(option_lines) + "\n请输入序号或完整代码。"
            
            # 循环直到用户输入有效
            while True:
                user_choice = interrupt({
                    "type": "multiple_matches",
                    "question": question_content,
                    "options": options
                })
                selected = parse_user_selection(user_choice, options)
                if selected:
                    break
                # 无效输入，更新提示内容，继续中断
                question_content = f"输入 '{user_choice}' 无效，请重新输入序号或完整代码。"
            
            # 用户选择成功，生成一条简短的确认消息加入历史
            confirm_msg = AIMessage(content=f"已选择：{selected['name']} ({selected['code']})，正在获取数据……")
            # 构造一条 HumanMessage，代表用户的选择结果，加入 messages 中供 LLM 理解
            user_selection_msg = HumanMessage(content=f"我选择：{selected['name']} ({selected['code']})")
            return {
                "stock_code": selected["code"],
                "stock_name": selected["name"],
                "conversation_history": [confirm_msg],   # ✅ 只写确认消息
                "messages": [user_selection_msg],   # 关键：更新 messages，让 LLM 知道选择已完成
            }

        elif "stock_code" in result and "stock_name" in result:
            return {
                "stock_code": result["stock_code"],
                "stock_name": result["stock_name"]
            }
    return {}


def parse_user_selection(user_input: str, options: list) -> dict | None:
    """解析用户的选择（支持序号或代码）。"""
    user_input = user_input.strip()
    # 尝试按序号匹配
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(options):
            return options[idx]
    # 尝试按代码匹配
    for opt in options:
        if opt["code"].endswith(user_input) or user_input == opt["code"]:
            return opt
    return None


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
    data_available = state.get("data_available", False)
    fetch_times = state.get("fetch_times", 0)

    # 数据已成功获取，结束研究员流程
    if data_available:
        print("[Researcher] 数据已就绪，流程结束。")
        return END

    # 数据未获取，检查是否还有重试机会
    if fetch_times < 5:
        # 如果还没有代码或名称，也需要继续（但此时可能由工具中断处理）
        if intent == "price_check" or intent == "analyze_only" or intent == "full_advice":
            # 如果有股票代码和名称但数据失败，可以重试 fetch_data
            if state.get("stock_code") and state.get("stock_name"):
                print("[Researcher] 数据采集失败，尝试重新调用 fetch_data")
                return "llm"
            # 否则可能还需要先确定股票，也返回 llm 让 LLM 决策
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