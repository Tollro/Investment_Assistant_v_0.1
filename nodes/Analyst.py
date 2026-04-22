"""
Analyst节点：客观分析股票信息（基本面、技术面、消息面、风险）
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
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
import os
import time


def print_messages_simple(messages):
    """简洁打印消息列表"""
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
    conversation_history: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    stock_code: Optional[str]
    stock_name: Optional[str]
    collected_data: Optional[Dict[str, Any]]
    data_available: bool
    analysis: str


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
    return llm


def call_llm_analysis(state: AnalystState, llm=None) -> dict:
    """
    调用 LLM 生成客观分析，存入 messages。
    若 analysis 字段已有内容且无需覆盖，可在此判断跳过（可选）。
    """
    if llm is None:
        raise ValueError("llm must be provided to call_llm_analysis.")

    # 如果 analysis 已有内容且希望跳过生成，可取消下面注释
    # if state.get("analysis"):
    #     return {}

    system_prompt = """
你是一名拥有二十年从业经验的资深股票分析师，风格客观严谨。
你的任务：基于提供的股票数据，对该股票进行全面、中立的分析，**仅分析事实，不提供任何投资建议**。

分析必须包含以下四个方面（若某方面数据缺失，请注明“数据暂缺”）：
1. **基本面分析**：盈利能力（ROE、毛利率）、成长性（营收/利润增速）、估值水平（PE/PB历史分位）、现金流与负债结构。
2. **技术面分析**：近期价格走势、关键均线位置、成交量变化、技术指标信号（如MACD、RSI等）。
3. **消息与事件面**：近期公告、行业政策、机构评级、新闻舆情倾向。
4. **风险提示**：经营风险、行业竞争、宏观政策影响等潜在负面因素。

要求：
- 每个观点必须基于给出的客观数据，不得主观臆断。
- 语言平实清晰，避免模糊表述。
- 使用中文输出，分段描述，不要使用Markdown表格或代码块。
- 仅分析，不提出“买入/卖出/持有”等建议。
"""

    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    stock_name = state.get("stock_name", "")
    stock_code = state.get("stock_code", "")
    collected_data = state.get("collected_data", {})

    if not messages:
        # 格式化数据
        data_str = json.dumps(collected_data, ensure_ascii=False, indent=2) if collected_data else "无数据"
        human_message = f"""
用户提问：{user_query}
股票名称：{stock_name}
股票代码：{stock_code}
已采集数据如下：
{data_str}

请根据以上信息，按要求的四个方面进行客观分析。
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ]
    else:
        if not isinstance(messages[0], SystemMessage):
            messages.insert(0, SystemMessage(content=system_prompt))

    response = llm.invoke(messages)
    return {"messages": [response]}


def update_analysis(state: AnalystState) -> dict:
    """
    从最后一条 AIMessage 中提取文本内容，更新 analysis 字段。
    """
    messages = state.get("messages", [])
    if not messages:
        return {"analysis": ""}

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        return {"analysis": last_msg.content.strip()}
    return {}


def Analyst_Graph() -> CompiledStateGraph:
    llm = create_llm(temperature=0.2)

    workflow = StateGraph(AnalystState)

    workflow.add_node("llm", lambda state: call_llm_analysis(state, llm))
    workflow.add_node("update_analysis", update_analysis)

    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", "update_analysis")
    workflow.add_edge("update_analysis", END)

    return workflow.compile()


if __name__ == "__main__":

    # 模拟已采集数据
    mock_collected_data = {
        "price": 1680.00,
        "pe": 28.5,
        "pb": 6.2,
        "roe": 24.3,
        "revenue_growth": 15.2,
        "profit_growth": 12.1,
        "debt_ratio": 0.22,
        "rsi_14": 58.3,
        "ma_50": 1650.0,
        "ma_200": 1520.0,
        "news_sentiment": "中性偏积极，近期多家机构维持买入评级，飞天茅台批价稳定。"
    }

    graph = Analyst_Graph()

    query = "我想查询茅台的相关信息"
    initial_state = {
        "user_query": query,
        "intent": "unknown",
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "collected_data": mock_collected_data,
        "analysis": ""                  # 初始为空，将由节点填充
    }

    start_time = time.time()
    result = graph.invoke(initial_state)
    end_time = time.time()
    print(f"\n========== 执行完成 ==========")
    print(f"单轮总耗时：{end_time - start_time:.4f} 秒")

    print("\n=== 最终分析内容 (analysis字段) ===")
    print(result["analysis"])
    print("\n=== 消息记录 ===")
    print_messages_simple(result["messages"])