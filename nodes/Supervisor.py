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

class SupervisorState(TypedDict):
    # ----- 全局状态----- 
    user_query: str
    intent: Literal["price_check", "analyze_only", "full_advice", "unknown"]
    
    # ----- Supervisor字段 -----
    last_worker: Literal["Supervisor", "Researcher", "Analyst", "Advisor"]         # 记录最后执行的子图名称，用于断点恢复
    next_worker: Optional[str]
    # needs_clarification: bool          # 是否需要暂停并向用户追问
    # clarification_question: str        # 向用户展示的追问内容
    # current_phase: Literal["collecting", "analyzing", "reporting", "interrupted"]

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
    "price_check": SentenceTransformer("查询股票实时价格、报价、涨跌幅"),
    "analyze_only": SentenceTransformer("分析股票走势、技术指标、K线形态，不需要投资建议"),
    "full_advice": SentenceTransformer("给出股票买入卖出建议、投资评级、持有还是抛售"),
}

# ---------- 辅助校验函数 ----------
# def validate_stock_code(code: str) -> bool:
#     """校验股票代码格式：交易所后缀（SH/SZ/BJ）+ 6位数字，如 sh600010"""
#     pattern = r"^(sh|sz|bj)\d{6}$"
#     return bool(re.match(pattern, code))

# def validate_collected_data(data: Dict) -> bool:
#     """检查 collected_data 是否包含最小必要字段"""
#     required_top_keys = {"stock_code", "collected_data"}
#     if not required_top_keys.issubset(data.keys()):
#         return False
#     market = data["collected_data"]["market_data"]
#     if "kline_daily" not in market or len(market["kline_daily"]) < 15:
#         return False
#     return True

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

def re_recognize_intent(query: str) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["价格", "报价", "多少钱", "实时"]):
        return "price_check"
    if any(kw in query_lower for kw in ["分析", "走势", "技术面"]):
        return "analyze_only"
    if any(kw in query_lower for kw in ["建议", "推荐", "买入", "卖出"]):
        return "full_advice"
    if "不" in query or "别" in query:
        return "unknown"
    return "unknown"

def vec_recognize_intent(query: str) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
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
    return best_intent if max_score > 0.5 else "unknown"

def llm_recognize_intent(query: str) -> Literal["price_check", "analyze_only", "full_advice", "unknown"]:
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
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,        # 消除随机性，保证输出稳定
                "num_predict": 20,          # 最大生成 token 数，防止废话
                "top_p": 0.1               # 极低核采样，增强确定性
            }
        )
        
        # 提取并清洗模型返回内容
        raw_output = response["message"]["content"].strip().lower()
        
        # 例如模型可能返回 "price_check。" 或 "intent: price_check"
        for label in ["price_check", "analyze_only", "full_advice", "unknown"]:
            if label in raw_output:
                return label
        
        # 如果没匹配到任何标签，记录日志并返回 unknown
        print(f"[LLM意图识别异常] 模型返回非预期内容: {raw_output}")
        return "unknown"
        
    except Exception as e:
        print(f"[LLM意图识别错误] 调用 Ollama 失败: {e}")
        return "unknown"  # 发生异常时安全兜底

# 混合识别，在特定情况下减小响应时间
def hybrid_recognize_intent(query: str) -> str:
    # 规则
    rule_result = re_recognize_intent(query)
    # 向量
    vec_result = vec_recognize_intent(query)

    if rule_result == vec_result and rule_result != "unknown":
        return vec_result
    else:
        # llm兜底
        return llm_recognize_intent(query)

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
    intent = state.get("intent")
    last_worker = state.get("last_worker")
    if intent and last_worker:
        if last_worker == "Supervisor":
            next_worker = "Researcher"
        elif last_worker == "Researcher":
            next_worker = "end" if intent == "price_check" else "Analyst"
        elif last_worker == "Analyst":
            next_worker = "end" if intent == "analyze_only" else "Advisor"
        elif last_worker == "Advisor":
            next_worker = "end"
        # elif last_worker == "start":
        #     next_worker = "Supervisor"
        else:
            print(f"\nSuperviorState中last_worker:{last_worker}出现错误！")
            return {}
        print(state)
        return {
            "next_worker": next_worker
        }
    return {}

def start_condition(state: SupervisorState) -> str:
    intent = state.get("intent", "")
    if not intent or intent == "unknown":
        return "get_intent"
    return "schedule"

def intent_condition(state: SupervisorState) -> str:
    intent = state.get("intent", "")
    # print(state)
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

    Supervisor = workflow.compile()

    return Supervisor

# def route_entry(state: SupervisorState) -> str:
#     # 如果处于中断状态且 messages 中有新的用户输入，直接让 Researcher 处理
#     if state.get("needs_clarification"):
#         return "researcher"
#     return "supervisor"


if __name__ == "__main__":
    Supervisor = Supervisor_Graph()
    initial_state = {
        # ----- 全局状态----- 
        "user_query": "对于茅台的股票由什么建议？",
        "intent": "analyze_only",
        
        # ----- Supervisor字段 -----
        "last_worker": "Researcher",
        "next_worker": ""
    }
    start_time = time.time()

    result = Supervisor.invoke(initial_state)
    end_time = time.time()
    print(f"\n========== 执行完成 ==========")
    print(f"单轮总耗时：{end_time - start_time:.4f} 秒")
    print(f"最终状态：{result}")
# builder.add_conditional_edges(START, route_entry)
