"""
Advisor节点：基于客观分析生成短中长期投资建议
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
import re
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


class AdvisorState(TypedDict):
    """Advisor Graph State"""
    messages: Annotated[List[BaseMessage], add_messages]

    # 同步全局 InvestmentState
    conversation_history: Annotated[List[BaseMessage], add_messages]
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]

    stock_code: Optional[str]
    stock_name: Optional[str]
    analysis: str                           # 从父图 Analyst 节点传入的客观分析
    advices: Dict[str, Any]                 # 存储短中长期建议的字典


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


def call_llm_advice(state: AdvisorState, llm=None) -> dict:
    """
    调用LLM，基于客观分析生成短、中、长期建议，要求以JSON格式返回。
    """
    if llm is None:
        raise ValueError("llm must be provided to call_llm_advice.")

    system_prompt = """
你是一名拥有二十年从业经验的资深投资顾问，擅长结合基本面、技术面和市场情绪给出分周期的投资建议。
你的任务：基于提供的客观分析报告，提炼出**短期、中期、长期**三个时间维度的前景判断与操作建议。

**输出要求**：
- 必须以严格的 **JSON格式** 返回，不得包含任何额外说明或Markdown标记（例如```json）。
- JSON结构必须包含以下字段，且字段名不可更改：
{
    "short_term": {
        "outlook": "短期（1-3个月）前景判断，简洁一句话",
        "reasoning": "判断依据，引用分析中的关键数据或事实",
        "suggestion": "短期操作建议（如增持、持有、观望、减持等）"
    },
    "mid_term": {
        "outlook": "中期（6-12个月）前景判断，简洁一句话",
        "reasoning": "判断依据，引用分析中的关键数据或事实",
        "suggestion": "中期操作建议"
    },
    "long_term": {
        "outlook": "长期（1-3年）前景判断，简洁一句话",
        "reasoning": "判断依据，引用分析中的关键数据或事实",
        "suggestion": "长期操作建议"
    }
}

**注意事项**：
1. 建议必须基于提供的客观分析，不能脱离数据凭空想象。
2. 不同周期的建议可以不同（例如短期谨慎、长期乐观），需自洽。
3. 语言专业、清晰，避免模糊表述。
4. 即使某些数据缺失，也要基于已有信息给出合理判断。
"""

    messages = state.get("messages", [])
    user_query = state.get("user_query", "")
    stock_name = state.get("stock_name", "")
    stock_code = state.get("stock_code", "")
    analysis = state.get("analysis", "")

    if not messages:
        human_message = f"""
用户提问：{user_query}
股票名称：{stock_name}
股票代码：{stock_code}

以下是专业分析师提供的客观分析报告：
---
{analysis if analysis else "（无现成分析，请回复'根据目前获得的信息无法给出有效的建议'）"}
---

请基于上述分析，生成符合要求的JSON格式建议。
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


def parse_advice_json(state: AdvisorState) -> dict:
    """
    从最后一条AIMessage中提取JSON，解析后存入advices字段。
    增强：不可见字符清理、正则提取、尾随逗号修复、详细日志。
    """
    messages = state.get("messages", [])
    if not messages:
        print("[parse_advice] 没有消息")
        return {"advices": {}}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        print(f"[parse_advice] 最后一条不是AIMessage: {type(last_msg)}")
        return {"advices": {}}

    raw_content = last_msg.content.strip()
    print(f"[parse_advice] 原始内容长度: {len(raw_content)}")

    # 1. 移除不可见字符（如 BOM、零宽空格）
    cleaned = raw_content.encode('utf-8').decode('utf-8-sig')  # 移除BOM
    cleaned = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', cleaned)  # 移除零宽字符

    # 2. 去除常见前缀干扰
    prefixes = [
        "以下是完整的json格式：", "这是生成的投资建议JSON：",
        "以下是JSON格式的建议：", "输出如下：", "JSON输出：",
        "以下为要求的JSON内容："
    ]
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):].strip()
            break

    # 3. 使用正则提取最外层 JSON 对象
    #    匹配从第一个 '{' 到对应的 '}'（处理嵌套）
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        print(f"[parse_advice] 正则提取成功，长度: {len(json_str)}")
    else:
        # 降级：手动查找首尾大括号
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start:end+1]
            print(f"[parse_advice] 手动提取成功，长度: {len(json_str)}")
        else:
            print("[parse_advice] 未找到 JSON 对象")
            return {"advices": {"raw": raw_content, "error": "No JSON object found"}}

    # 4. 去除可能残留的 Markdown 标记
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    # 5. 修复常见 JSON 错误：尾随逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # 6. 尝试解析
    try:
        advices_dict = json.loads(json_str)
        print("[parse_advice] JSON 解析成功")
    except json.JSONDecodeError as e:
        print(f"[parse_advice] JSON 解析失败: {e}")
        # 返回原始内容和错误信息
        return {"advices": {"raw": raw_content, "error": str(e)}}

    # 7. 验证必需字段
    required_keys = {"short_term", "mid_term", "long_term"}
    if not required_keys.issubset(advices_dict.keys()):
        print(f"[parse_advice] 缺少必需字段，现有字段: {list(advices_dict.keys())}")
        return {"advices": {
            "raw": raw_content,
            "error": "Missing required fields",
            "parsed": advices_dict
        }}

    print("[parse_advice] 成功更新 advices")
    return {"advices": advices_dict}


def Advisor_Graph() -> CompiledStateGraph:
    llm = create_llm(temperature=0.2)

    workflow = StateGraph(AdvisorState)

    workflow.add_node("llm", lambda state: call_llm_advice(state, llm))
    workflow.add_node("parse_advice", parse_advice_json)

    workflow.add_edge(START, "llm")
    workflow.add_edge("llm", "parse_advice")
    workflow.add_edge("parse_advice", END)

    return workflow.compile()


if __name__ == "__main__":
    mock_analysis = """..."""  # 你的模拟分析文本

    graph = Advisor_Graph()
    initial_state = {
        "user_query": "茅台投资建议",
        "intent": "full_advice",
        "stock_code": "600519",
        "stock_name": "贵州茅台",
        "collected_data": None,
        "analysis": mock_analysis,
        "advices": {}
    }

    start_time = time.time()
    result = graph.invoke(initial_state)
    end_time = time.time()
    print(f"\n========== 执行完成 ==========")
    print(f"单轮总耗时：{end_time - start_time:.4f} 秒")

    print("\n=== 最终建议 (advices) ===")
    print(json.dumps(result["advices"], ensure_ascii=False, indent=2))

    # 仅打印一次消息
    print("\n=== 消息记录 ===")
    print_messages_simple(result["messages"])