"""
Responder节点：根据历史消息生成回复，并重置状态
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated, Union
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
import os

WorkerType = Literal["Supervisor", "Researcher", "Analyst", "Advisor"]

class ResponderState(TypedDict):
    """Responder Graph State"""
    # 全局状态
    conversation_history: Annotated[List[BaseMessage], add_messages]
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
    # ----- Supervisor 流程控制字段 -----
    needs_clarification: bool
    clarification_question: str
    current_phase: Literal["collecting", "analyzing", "reporting", "interrupted"]
    
    last_worker: Optional[WorkerType]
    next_worker: Optional[Union[WorkerType, Literal["__end__"]]]

    # ----- 股票标识 -----
    stock_code: Optional[str]
    stock_name: Optional[str]

    # ----- 原始采集数据 -----
    collected_data: Optional[Dict[str, Any]]
    data_available: bool

    # ----- 中间分析结果 -----
    analysis: str
    advices: Dict[str, Any]              # Advisor 节点输出
    response: str                        # Responder 节点输出


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
    return llm


def call_llm_response(state: ResponderState, llm=None) -> dict:
    """
    调用 LLM 生成面向最终用户的回复。
    """
    if llm is None:
        raise ValueError("llm must be provided to call_llm_response.")

    user_query = state.get("user_query", "")
    stock_name = state.get("stock_name", "该股票")
    intent = state.get("intent", "unknown")
    analysis = state.get("analysis", "")
    advices = state.get("advices", {})

    # ----- 系统提示设计 -----
    system_prompt = """
你是一名专业的金融客服助手，负责向用户清晰、准确地传达股票分析和投资建议。
你的回复将直接展示给用户，请确保语言专业、友善、条理清晰。

**回复原则**：
1. 根据用户的问题意图调整回复重点：
   - 若用户仅询问价格（price_check），需告知最新价格、涨跌幅、最近一个月的周线数据、月线数据(如果用户没要求默认给月线数据)还有财务信息，无需展开分析。
   - 若用户要求分析（analyze_only），应客观陈述给出的分析中的各情况，**不主动提供买卖建议**。
   - 若用户寻求投资建议（full_advice），可结合专业分析给出短中长期的操作参考，但需添加风险提示。
2. 回复结构建议：
   - 开头：确认股票名称，回应问题核心。
   - 主体：根据意图展示相关内容（价格、分析要点、建议等）。
   - 结尾：可附加一句免责声明（如“以上分析仅供参考，不构成投资建议”）。
3. 避免输出未提供的信息，不编造数据。若某些信息缺失，诚实告知用户。
4. 语气亲切自然，适当使用“您”等敬语。
5. 输出纯文本，不使用 Markdown 表格或代码块。
6. 若没有任何数据，直接回复"抱歉，暂时找不到任何信息"
"""

    # ----- 构造人类消息（包含具体数据）-----
    # 根据意图定制数据呈现
    if intent == "price_check":
        data_section = f"""
        相关价格数据：{state.get('collected_data', {}).get('market_data', {})} 
        相关财务报表：{state.get('collected_data', {}).get("financial_reports", {})}
        """
    else:
        data_section = f"""
专业分析报告：
{analysis if analysis else "（暂无详细分析）"}

专业投资建议（JSON格式）：
{advices if advices else "（暂无具体建议）"}
"""

    human_content = f"""
用户提问：{user_query}
股票名称：{stock_name}（{state.get('stock_code', '')}）
用户意图：{intent}

以下是您需要参考的资料：
{data_section}

请根据上述信息生成对用户的最终回复。
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]

    # 调用 LLM
    response_msg = llm.invoke(messages)
    reply_text = response_msg.content.strip()

    # 更新 conversation_history 和 response / final_response 字段
    return {
        "messages": [response_msg],
        "conversation_history": [response_msg],
        "response": reply_text,
        "next_worker": "__end__"           # 标记流程结束，便于父图调度
    }


def Responder_Graph() -> CompiledStateGraph:
    llm = create_llm(temperature=0.2)

    workflow = StateGraph(ResponderState)

    workflow.add_node("llm", lambda state: call_llm_response(state, llm))

    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", END)

    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    # 简单测试示例
    graph = Responder_Graph()
    test_state = {
        "user_query": "茅台能买吗？",
        "intent": "full_advice",
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "analysis": "贵州茅台基本面稳健，ROE长期维持在30%以上，近期股价在1600-1700元区间震荡。",
        "advices": {
            "short_term": {"suggestion": "观望", "reasoning": "短期技术指标偏弱"},
            "mid_term": {"suggestion": "持有", "reasoning": "消费复苏预期"},
            "long_term": {"suggestion": "增持", "reasoning": "品牌护城河深厚"}
        },
        "conversation_history": [],
        "needs_clarification": False,
        "clarification_question": "",
        "current_phase": "reporting",
        "last_worker": "Advisor",
        "next_worker": None,
        "collected_data": {"market_data": {"realtime": {"price": 1680.00}}},
        "data_available": True,
        "response": "",
        "error_info": None,
        "retry_count": 0,
        "final_response": None
    }
    config = {"configurable": {"thread_id": "test_responder"}}
    result = graph.invoke(test_state, config)
    print("\n===== 最终回复 =====\n")
    print(result.get("final_response"))