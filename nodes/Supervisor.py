"""
Supervisor节点：调度/管理/分配任务
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import re
from langgraph.graph import add_messages, StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from typing import Literal, Dict, Union, List, Optional, TypedDict, Any, Annotated
from langchain_core.messages import BaseMessage
import requests
from tqdm import tqdm
import time
import numpy as np
import ollama


WorkerType = Literal["Supervisor", "Researcher", "Analyst", "Advisor"]

class SupervisorState(TypedDict):
    # ----- 全局状态----- 
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
    # ----- Supervisor字段 -----
    last_worker: Optional[WorkerType]          # 记录最后执行的子图
    next_worker: Optional[Union[WorkerType, Literal["end"]]]  # 下一步去向
    # needs_clarification: bool          # 是否需要暂停并向用户追问
    # clarification_question: str        # 向用户展示的追问内容
    # current_phase: Literal["collecting", "analyzing", "reporting", "interrupted"]

    # 可扩展字段
    stock_code: Optional[str]
    stock_name: Optional[str]
    data_available: bool


def SentenceTransformer(texts: Union[str, List[str]], batch_size: int = 32) -> Union[List[float], List[List[float]]]:
    """
    获取文本的语义向量，支持单句或批量输入。
    
    Args:
        texts: 单个字符串或字符串列表
        batch_size: 批量处理时每批的大小
    
    Returns:
        单句输入 -> List[float]  (一个向量)
        批量输入 -> List[List[float]] (多个向量)
    """
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]
    
    all_embeddings = []
    url = "http://localhost:11434/api/embed"
    total = len(texts)
    
    with tqdm(total=total, desc="语义向量化进度", unit="条", disable=(total == 1)) as pbar:
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
                print(f"批量请求失败 (start_idx={start_idx}): {e}")
                raise
    
    # 单句输入时返回单个向量，不再嵌套列表
    return all_embeddings[0] if single_input else all_embeddings


# ----------- 引入语义向量模型 -----------
INTENT_EMBEDDINGS = {
    "price_check": SentenceTransformer("查询股票实时价格或报价或涨跌幅"),
    "analyze_only": SentenceTransformer("分析股票走势或技术指标或K线形态，不需要投资建议"),
    "full_advice": SentenceTransformer("给出股票买入卖出建议或投资评级或持有还是抛售"),
}


# 从文本中简单提取股票代码
def extract_stock_code(query: str) -> Optional[str]:
    import re
    # 匹配 6 位数字代码，并自动补全前缀（默认按上交所处理，实际需智能判断）
    match = re.search(r"\b(\d{6})\b", query)
    if match:
        code = match.group(1)
        if code.startswith(("600", "601", "603", "605")):
            return f"sh{code}"
        elif code.startswith(("000", "002", "003", "300")):
            return f"sz{code}"
        elif code.startswith(("430", "830", "831")):
            return f"bj{code}"
        else:
            return f"sh{code}"  # 默认赋予上交所前缀
    return None


def rule_based_intent(query: str) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
    """基于规则的意图识别，快速响应常见关键词。"""
    query_lower = query.lower()
    # 价格查询关键词
    if any(kw in query_lower for kw in ["价格", "报价", "多少钱", "实时", "现价", "涨跌幅", "涨了", "跌了"]):
        return "price_check"
    # 分析关键词（不含建议）
    if any(kw in query_lower for kw in ["分析", "走势", "技术面", "k线", "指标", "形态", "趋势"]):
        return "analyze_only"
    # 建议关键词
    if any(kw in query_lower for kw in ["建议", "推荐", "买入", "卖出", "持有", "抛售", "操作", "策略"]):
        return "full_advice"
    return "unknown"


def vector_based_intent(query: str, threshold=0.5 ) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
    query_vec = SentenceTransformer(query)
    max_score = -1
    best_intent = "unknown"
    for intent, vec in INTENT_EMBEDDINGS.items():
        # 计算余弦相似度
        score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        if score > max_score:
            max_score = score
            best_intent = intent
            
    # 设定阈值，低于阈值依然归为 unknown 触发中断
    return best_intent if max_score > threshold else "unknown"


def clean_llm_output(raw: str) -> str:
    """
    清洗 LLM 输出，移除 DeepSeek-R1 的  标签及其他干扰内容。
    """
    # 移除  ...  标签及其内容
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # 移除可能残留的 XML 标签
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # 提取第一个出现的有效意图标签
    for label in ["price_check", "analyze_only", "full_advice", "unknown"]:
        if label in cleaned.lower():
            return label
    return "unknown"


def llm_based_intent(query: str) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
    """
    使用 Ollama 本地模型进行意图识别兜底。
    强制模型仅返回四个预定义标签之一。
    """
    # 使用更强的指令约束输出格式
    prompt = f"""你是一个金融意图分类器。请严格遵循以下规则：
    【可选意图标签】
    - price_check      : 用户只想查询实时价格、报价、涨跌幅
    - analyze_only     : 用户要求分析走势、技术指标、K线形态，不需要投资建议
    - full_advice      : 用户要求给出买卖建议、操作策略、持有或抛售的判断
    - unknown          : 无法明确归类到以上三种意图

    【输出要求】
    仅输出四个标签中的一个，不要输出任何解释、空格、标点或换行。

    【用户输入】
    {query}

    【意图标签】
    """
    try:
        print("\n使用llm获取用户intent......")
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,        # 消除随机性，保证输出稳定
                # "num_predict": 50,          # 最大生成 token 数，防止废话
                "top_p": 0.1               # 极低核采样，增强确定性
            }
        )
        # 提取并清洗模型返回内容
        raw_output = response["message"]["content"].strip().lower()
        print(f"[LLM原始输出] {raw_output[:100]}...")  # 调试用
        cleaned = clean_llm_output(raw_output)
        print(f"[LLM清洗后] {cleaned}")
        return cleaned
    except Exception as e:
        print(f"[LLM意图识别错误] {e}")
        return "unknown"

# 混合识别，在特定情况下减小响应时间
def hybrid_recognize_intent(query: str) -> str:
    # 规则
    rule_result = rule_based_intent(query)
    # 向量
    vec_result = vector_based_intent(query, threshold=0.5)

    if rule_result == vec_result and rule_result != "unknown":
        return vec_result
    else:
        # llm兜底
        return llm_based_intent(query)


def get_intent_node(state: SupervisorState) -> dict:
    print("\n进入get_intent_node... ...")
    query = state.get("user_query", "")
    if query != "":
        intent = hybrid_recognize_intent(query)
        print(f"\n提取到用户意图：{intent}")
        return {"intent": intent}
    print(f"\n!用户意图：unknown")
    return {"intent": "unknown"}


def schedule_node(state: SupervisorState) -> dict:
    print("\n进入schedule_node... ...")
    intent = state.get("intent", "unknown")
    last = state.get("last_worker")
    data_available = state.get("data_available", False)

    # 初始进入：last 可能为 None，表示刚从 Supervisor 开始
    if last is None:
        next_worker = "Researcher"
        print(f"[Supervisor] 初始调度 -> Researcher")
        return {"next_worker": next_worker, "last_worker": "Supervisor"}

    # 调度表：基于 (intent, last_worker) 映射到下一个 worker 或结束
    # 格式: (intent, last) -> next
    schedule_map = {
        ("price_check", "Researcher"): "end",
        ("analyze_only", "Researcher"): "Analyst" if data_available else "end",
        ("full_advice", "Researcher"): "Analyst" if data_available else "end",
        ("analyze_only", "Analyst"): "end",
        ("full_advice", "Analyst"): "Advisor",
        ("full_advice", "Advisor"): "end",
    }

    # 特殊：若意图 unknown，则直接结束
    if intent == "unknown":
        print("[Supervisor] 意图未知，流程结束")
        return {"next_worker": "end"}

    key = (intent, last)
    next_worker = schedule_map.get(key)
    if next_worker is None:
        # 默认结束
        print(f"[Supervisor] 未找到调度规则 (intent={intent}, last={last})，默认结束")
        next_worker = "end"

    print(f"[Supervisor] 调度: last={last}, intent={intent} -> next={next_worker}")
    return {"next_worker": next_worker}


# ================= 条件路由 =================
def start_condition(state: SupervisorState) -> str:
    """START 分支：若已有意图且不为 unknown，直接进入调度；否则先识别意图。"""
    intent = state.get("intent")
    if intent and intent != "unknown":
        return "schedule"
    return "get_intent"


def intent_condition(state: SupervisorState) -> str:
    """识别意图后：若仍为 unknown，结束流程；否则进入调度。"""
    intent = state.get("intent", "unknown")
    if intent == "unknown":
        return END
    return "schedule"


def Supervisor_Graph() -> CompiledStateGraph:
    """
    监督者节点：负责任务编排。
    注意：该节点不直接执行任何业务计算，仅做决策和状态更新。
    """
    workflow = StateGraph(SupervisorState)

    workflow.add_node("get_intent", get_intent_node)
    workflow.add_node("schedule", schedule_node)

    workflow.add_conditional_edges(
        START,
        start_condition
    )
    workflow.add_conditional_edges(
        "get_intent",
        intent_condition
    )
    workflow.add_edge("schedule", END)

    return workflow.compile()


if __name__ == "__main__":
    Supervisor = Supervisor_Graph()
    test_queries = [
        "茅台现在多少钱？",
        "帮我分析一下茅台的走势",
        "茅台可以买入吗？"
    ]
    for q in test_queries:
        print(f"\n{'='*40}\n用户输入: {q}")
        initial_state = {
            "user_query": q,
            "intent": "",           # 初始为空，触发意图识别
            "last_worker": None,
            "next_worker": "",
            "data_available": False
        }
        start_time = time.time()
        result = Supervisor.invoke(initial_state)
        end_time = time.time()
        print(f"识别意图: {result.get('intent')}")
        print(f"下一步: {result.get('next_worker')}")
        print(f"\n========== 执行完成 ==========")
        print(f"单轮总耗时：{end_time - start_time:.4f} 秒")
        print(f"最终状态：{result}")
# builder.add_conditional_edges(START, route_entry)
