"""
Analyst Agent 演示程序
- 仅处理 State 中已有的 market_data 与 fundamental_data
- 提供三个专属工具：技术指标计算、估值判断、情绪摘要
- 无网络权限，工具均为本地模拟
"""

from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from dataclasses import dataclass
import random

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from test.model import create_gpt_call


# ==================== State 定义 ====================
class GraphState(TypedDict):
    """
    Analyst Agent 的状态结构。
    假设 Researcher 已填充 market_data 与 fundamental_data。
    """
    messages: Annotated[List[BaseMessage], add_messages]
    # 结构化收集的数据
    collected_data: Optional[Dict[str, Any]]


# ==================== 模拟数据工具 ====================
class TechnicalIndicatorsInput(BaseModel):
    """技术指标计算输入（从 State 中自动获取 OHLCV，无需传参）"""
    pass  # 工具内部直接从 State 读取，此处留空


class ValuationInput(BaseModel):
    """估值判断输入"""
    pass


class SentimentInput(BaseModel):
    """情绪摘要输入"""
    pass


@tool(args_schema=TechnicalIndicatorsInput)
def calculate_technical_indicators() -> str:
    """
    基于 OHLCV 数据计算 MACD、RSI、布林带位置及均线排列状态。
    实际实现中需从 State 获取 market_data，此处为模拟返回。
    """
    # 在真实场景中，可以从全局状态或上下文获取 market_data
    # 这里返回模拟结果
    indicators = {
        "MACD": "金叉，多头信号",
        "RSI": 58.3,
        "Bollinger_position": "中轨上方，接近上轨",
        "MA_arrangement": "多头排列"
    }
    return f"技术指标计算结果：{indicators}"


@tool(args_schema=ValuationInput)
def evaluate_valuation() -> str:
    """
    基于 PE、PB 及行业均值输出估值分位数判断（低估/合理/高估）。
    """

    valuation = {
        "PE": 12.5,
        "PE_industry_avg": 15.2,
        "PB": 1.8,
        "PB_industry_avg": 2.1,
        "verdict": "低估"
    }
    return f"估值分析结果：{valuation}"


@tool(args_schema=SentimentInput)
def summarize_sentiment() -> str:
    """
    输入新闻标题列表，调用轻量级 NLP 模型输出情绪倾向及关键词云。
    此处为模拟实现。
    """
    sentiment = {
        "overall": "positive",
        "confidence": 0.78,
        "keywords": ["业绩增长", "新产品发布", "机构增持"]
    }
    return f"情绪分析结果：{sentiment}"


# ==================== 错误处理中间件 ====================
def handle_tool_errors(request, handler):
    """统一工具错误处理，返回友好的错误消息。"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"工具执行出错：{str(e)}",
            tool_call_id=request.tool_call["id"]
        )


# ==================== 构建 Agent ====================
def create_analyst_agent(model: AzureChatOpenAI, tools: List, checkpointer=None):
    """
    使用 LangGraph 的 create_agent 快速构建 Analyst Agent。
    """
    from langchain.agents import create_agent

    system_prompt = """
你是一名专业的股票分析师（Analyst Agent）。
你的职责是基于已经提供的市场数据（market_data）、基本面数据（fundamental_data）
以及新闻标题（news_titles）进行定量计算与定性解读。
你能调用以下专属工具：
- calculate_technical_indicators：计算技术指标
- evaluate_valuation：评估估值水平
- summarize_sentiment：分析新闻情绪

当用户要求分析时，请按顺序调用上述工具，然后整合结果给出专业的分析报告。
"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[handle_tool_errors],
        state_schema=AnalystState,
        checkpointer=checkpointer
    )
    return agent


# ==================== 辅助函数：打印消息流程 ====================
def print_messages_simple(messages):
    """简洁打印消息历史，便于观察 Agent 推理与工具调用。"""
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
            print(f"内容: {msg.content}")


# ==================== 演示入口 ====================
if __name__ == "__main__":

    llm = create_gpt_call(temperature=0.2, max_tokens=800)

    # 准备 Analyst 专属工具
    tools = [
        calculate_technical_indicators,
        evaluate_valuation,
        summarize_sentiment
    ]

    # 创建带记忆的 Agent
    memory = MemorySaver()
    agent = create_analyst_agent(llm, tools, checkpointer=memory)

    # 模拟 Researcher 已经写入的状态数据
    initial_state = {
        "messages": [HumanMessage(content="请分析股票 600012 的技术面、估值和情绪面，并给出综合研判。")],
        # 以下数据在真实场景中由 Researcher Agent 填入
        "market_data": {
            "symbol": "sh600012",
            "ohlcv": [...]  # 省略具体数据
        },
        "fundamental_data": {
            "pe": 12.5,
            "pb": 1.8,
            "industry_pe_avg": 15.2,
            "industry_pb_avg": 2.1
        },
        "news_titles": [
            "公司一季度净利润同比增长20%",
            "机构调研频繁，看好下半年需求复苏",
            "新产品获得重要认证"
        ]
    }

    # 配置会话线程 ID
    session_config = {"configurable": {"thread_id": "user_001"}}

    print("\n" + "=" * 60)
    print("Analyst Agent 开始分析 (模拟数据)")
    print("=" * 60)

    # 运行 Agent
    result = agent.invoke(initial_state, config=session_config)

    # 打印完整交互记录
    print("\n" + "=" * 60)
    print("执行流程详细记录")
    print("=" * 60)
    print_messages_simple(result["messages"])