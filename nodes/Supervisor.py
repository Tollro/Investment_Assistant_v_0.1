"""
Supervisor节点：调度/管理/分配任务（优化版）
"""
import sys
from pathlib import Path
import re
import time
import numpy as np
import requests
import ollama
from tqdm import tqdm

from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from typing import Literal, Optional, Union, List, Dict, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ---------- 类型定义 ----------
WorkerType = Literal["Supervisor", "Researcher", "Analyst", "Advisor"]

class SupervisorState(TypedDict):
    # 全局字段
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    conversation_history: Annotated[List[BaseMessage], add_messages]
    
    # 调度控制
    last_worker: Optional[WorkerType]          # 上一次执行的子图（Supervisor 自身不算）
    next_worker: Optional[Union[WorkerType, Literal["__end__"]]]
    
    # 业务数据
    stock_code: Optional[str]
    stock_name: Optional[str]
    data_available: bool


# ---------- 语义向量工具（保持不变）----------
def SentenceTransformer(texts: Union[str, List[str]], batch_size: int = 32) -> Union[List[float], List[List[float]]]:
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]
    all_embeddings = []
    url = "http://localhost:11434/api/embed"
    total = len(texts)
    with tqdm(total=total, desc="语义向量化", unit="条", disable=(total == 1)) as pbar:
        for start_idx in range(0, total, batch_size):
            batch = texts[start_idx:start_idx + batch_size]
            payload = {"model": "bge-m3", "input": batch}
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend(result["embeddings"])
                pbar.update(len(batch))
            except Exception as e:
                print(f"批量请求失败: {e}")
                raise
    return all_embeddings[0] if single_input else all_embeddings

INTENT_EMBEDDINGS = {
    "price_check": SentenceTransformer("查询股票实时价格或报价或涨跌幅"),
    "analyze_only": SentenceTransformer("分析股票走势或技术指标或K线形态，股票分析，不需要投资建议"),
    "full_advice": SentenceTransformer("给出股票买入卖出建议或投资评级或持有还是抛售"),
}

# ---------- 意图识别（保持不变）----------
def rule_based_intent(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["查询", "价格", "报价", "多少钱", "实时", "现价", "涨跌幅", "涨了", "跌了"]):
        return "price_check"
    if any(kw in q for kw in ["分析", "走势", "技术面", "k线", "指标", "形态", "趋势"]):
        return "analyze_only"
    if any(kw in q for kw in ["建议", "推荐", "买入", "卖出", "持有", "抛售", "操作", "策略"]):
        return "full_advice"
    return "unknown"

def vector_based_intent(query: str, threshold=0.5) -> str:
    query_vec = SentenceTransformer(query)
    best, best_score = "unknown", -1
    for intent, vec in INTENT_EMBEDDINGS.items():
        score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        if score > best_score:
            best_score, best = score, intent
    return best if best_score > threshold else "unknown"

def clean_llm_output(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    for label in ["price_check", "analyze_only", "full_advice", "unknown"]:
        if label in cleaned.lower():
            return label
    return "unknown"


def llm_based_intent(query: str) -> str:
    prompt = f"""你是金融意图分类器，仅输出以下四个标签之一：
- price_check：查询价格/报价/涨跌幅
- analyze_only：分析走势/技术指标，不需要建议
- full_advice：买卖建议/操作策略
- unknown：其他

用户输入：{query}
标签："""
    try:
        print("使用 LLM 获取意图...")
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "top_p": 0.05}
        )
        raw = response["message"]["content"].strip().lower()
        print(f"[[LLM原始消息]\n{raw}")
        return clean_llm_output(raw)
    except Exception as e:
        print(f"[LLM错误] {e}")
        return "unknown"


def hybrid_recognize_intent(query: str) -> str:
    rule = rule_based_intent(query)
    vec = vector_based_intent(query, threshold=0.5)
    if rule == vec and rule != "unknown":
        return rule
    return llm_based_intent(query)


# ---------- Supervisor 节点函数（优化核心）----------

def update_last_worker_node(state: SupervisorState) -> dict:
    """
    推断上一次实际执行的 worker。
    依据：父图每次执行完子图都会回到 Supervisor，此时 state 中 next_worker 仍保留着
    上一轮调度时设置的值。我们可将该值作为 last_worker。
    特殊情况：刚启动时 next_worker 为空，则 last_worker 为 None。
    """
    next_val = state.get("next_worker")
    # 如果 next_worker 是有效子图名，说明它刚刚被执行完毕，将其记录为 last_worker
    if next_val in ("Researcher", "Analyst", "Advisor"):
        last = next_val
    else:
        # 否则可能是初始状态或结束标记，保留原有 last_worker 或设为 None
        last = state.get("last_worker")
    print(f"[Supervisor] 更新 last_worker: {last}")
    return {"last_worker": last}


def get_intent_node(state: SupervisorState) -> dict:
    query = state.get("user_query", "")
    if not query:
        return {"intent": "unknown"}
    intent = hybrid_recognize_intent(query)
    print(f"[Supervisor] 识别意图: {intent}")
    return {"intent": intent}


def schedule_node(state: SupervisorState) -> dict:
    """
    根据意图和 last_worker 决定下一步。
    调度规则表清晰且完整，包含错误处理路径。
    """
    intent = state.get("intent", "unknown")
    last = state.get("last_worker")
    data_available = state.get("data_available", False)

    # 若意图未知，直接结束
    if intent == "unknown":
        print("[Supervisor] 意图未知，转交Responder处理")
        return {"next_worker": "Responder"}

    # 定义调度表：(intent, last_worker) -> next_worker
    # 注意：last_worker 为 None 或 "Supervisor" 都表示初始状态
    schedule_map = {
        # 价格查询流程
        ("price_check", None): "Researcher",
        ("price_check", "Supervisor"): "Researcher",
        ("price_check", "Researcher"): "Responder",

        # 仅分析流程
        ("analyze_only", None): "Researcher",
        ("analyze_only", "Supervisor"): "Researcher",
        ("analyze_only", "Researcher"): "Analyst" if data_available else "Researcher",
        ("analyze_only", "Analyst"): "Responder",

        # 完整建议流程
        ("full_advice", None): "Researcher",
        ("full_advice", "Supervisor"): "Researcher",
        ("full_advice", "Researcher"): "Analyst" if data_available else "Researcher",
        ("full_advice", "Analyst"): "Advisor",
        ("full_advice", "Advisor"): "Responder",
    }

    key = (intent, last)
    next_worker = schedule_map.get(key)

    # 若未命中规则，尝试将 last 视为 "Supervisor" 重试（兜底）
    if next_worker is None and last not in (None, "Supervisor"):
        fallback_key = (intent, "Supervisor")
        next_worker = schedule_map.get(fallback_key)

    if next_worker is None:
        print(f"[Supervisor] 无调度规则 (intent={intent}, last={last})，默认结束")
        next_worker = "Responder"

    print(f"[Supervisor] 调度: last={last}, intent={intent}, data_available={data_available} -> next={next_worker}")
    return {"next_worker": next_worker}


# ---------- 条件路由 ----------
def should_get_intent(state: SupervisorState) -> str:
    """判断是否需要先识别意图"""
    intent = state.get("intent")
    if intent and intent != "unknown":
        return "schedule"
    return "get_intent"


def Supervisor_Graph() -> CompiledStateGraph:
    workflow = StateGraph(SupervisorState)

    workflow.add_node("update_last", update_last_worker_node)
    workflow.add_node("get_intent", get_intent_node)
    workflow.add_node("schedule", schedule_node)

    # 入口：先更新 last_worker
    workflow.add_edge(START, "update_last")

    # 根据意图是否已知决定路径
    workflow.add_conditional_edges(
        "update_last",
        should_get_intent,
        {"get_intent": "get_intent", "schedule": "schedule"}
    )
    workflow.add_edge("get_intent", "schedule")
    workflow.add_edge("schedule", END)

    return workflow.compile()


# ---------- 测试 ----------
if __name__ == "__main__":
    supervisor = Supervisor_Graph()
    test_cases = [
        {"query": "茅台现在多少钱？", "intent": "", "last_worker": None, "data_available": False},
        {"query": "分析茅台走势", "intent": "", "last_worker": None, "data_available": False},
        {"query": "茅台能买吗？", "intent": "", "last_worker": None, "data_available": False},
    ]
    for case in test_cases:
        state = {
            "user_query": case["query"],
            "intent": case["intent"],
            "last_worker": case["last_worker"],
            "next_worker": "",
            "data_available": case["data_available"],
            "conversation_history": [],
        }
        print("\n" + "="*50)
        print(f"用户: {case['query']}")
        result = supervisor.invoke(state)
        print(f"意图: {result.get('intent')}")
        print(f"last_worker: {result.get('last_worker')}")
        print(f"next_worker: {result.get('next_worker')}")