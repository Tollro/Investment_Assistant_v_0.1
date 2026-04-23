import streamlit as st
from langgraph.types import Command
from langchain_core.messages import HumanMessage
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from nodes.Stock_Graph_Single import Stock_Graph_Single

st.set_page_config(page_title="金融助手", page_icon="📈", layout="wide")
st.title("📈 金融助手")
st.caption("基于 LangGraph 的多智能体协作系统 —— 支持股票查询、分析与建议")

# ---------- 初始化会话状态 ----------
if "assistant" not in st.session_state:
    st.session_state.assistant = Stock_Graph_Single()
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "ui_session_001"}}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interrupt_event" not in st.session_state:
    st.session_state.interrupt_event = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "user_input_to_process" not in st.session_state:
    st.session_state.user_input_to_process = None

# ---------- 辅助函数 ----------
def run_graph_and_get_response(user_input_text, is_resume=False):
    """执行图，返回 (是否中断, 最终回复)"""
    assistant = st.session_state.assistant
    config = st.session_state.config

    if is_resume:
        input_state = Command(resume=user_input_text)
    else:
        input_state = {
            "user_query": user_input_text,
            "conversation_history": [HumanMessage(content=user_input_text)]
        }

    events = assistant.stream(input_state, config)
    final_response = None
    interrupted = False

    for event in events:
        if "__interrupt__" in event:
            interrupt_info = event["__interrupt__"][0]
            st.session_state.interrupt_event = {
                "prompt": interrupt_info.value["question"],
                "resume_needed": True
            }
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❓ {interrupt_info.value['question']}"
            })
            interrupted = True
            break

        for node_name, node_output in event.items():
            if isinstance(node_output, dict) and "response" in node_output:
                candidate = node_output["response"]
                if candidate and isinstance(candidate, str) and candidate.strip():
                    final_response = candidate
                    # 同时捕获分析数据
                    if "analysis" in node_output:
                        st.session_state.last_analysis = node_output["analysis"]
                    if "advices" in node_output:
                        st.session_state.last_advices = node_output["advices"]
                    if "stock_name" in node_output:
                        st.session_state.last_stock_name = node_output["stock_name"]

    if not final_response and not interrupted:
        final_state = assistant.get_state(config).values
        final_response = final_state.get("response", "")

    return interrupted, final_response


# ---------- 渲染对话历史 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 处理待执行的用户输入 ----------
if st.session_state.processing and st.session_state.user_input_to_process is not None:
    prompt = st.session_state.user_input_to_process
    is_resume = st.session_state.interrupt_event is not None

    # 立即将用户消息加入历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 添加一个临时助手消息
    temp_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": "⏳ 正在分析..."})

    # 清空处理标志
    st.session_state.processing = False
    st.session_state.user_input_to_process = None

    # 刷新界面显示用户消息和临时消息
    st.rerun()

# ---------- 处理输入框 ----------
if st.session_state.interrupt_event is not None:
    prompt_placeholder = st.session_state.interrupt_event["prompt"]
    with st.chat_message("assistant"):
        st.info(f"⏸️ 等待您的回答：{prompt_placeholder}")
else:
    prompt_placeholder = "请输入您的问题，例如：分析一下贵州茅台"

user_input = st.chat_input(prompt_placeholder)

if user_input:
    # 设置处理标志，将在下一次渲染时执行
    st.session_state.processing = True
    st.session_state.user_input_to_process = user_input
    st.rerun()

# ---------- 执行图并更新临时消息 ----------
if "temp_index" in locals():
    temp_index = locals()["temp_index"]
else:
    # 如果当前消息列表最后一条是临时消息，则处理它
    if st.session_state.messages and st.session_state.messages[-1]["content"] == "⏳ 正在分析...":
        temp_index = len(st.session_state.messages) - 1
    else:
        temp_index = None

if temp_index is not None:
    # 取出用户问题（上一条消息）
    if temp_index > 0:
        user_msg = st.session_state.messages[temp_index - 1]["content"]
        is_resume = st.session_state.interrupt_event is not None

        # 执行图
        interrupted, final_response = run_graph_and_get_response(user_msg, is_resume)

        if not interrupted and final_response:
            # 替换临时消息为最终回复
            st.session_state.messages[temp_index]["content"] = final_response
        elif interrupted:
            # 中断情况：临时消息已被中断提示替换，无需额外操作
            pass
        else:
            # 无回复，移除临时消息
            st.session_state.messages.pop(temp_index)

        # 刷新界面
        st.rerun()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📋 最新分析报告")
    if st.session_state.get("last_stock_name"):
        st.subheader(f"📌 {st.session_state.last_stock_name}")
    if st.session_state.get("last_analysis"):
        with st.expander("🔍 详细分析", expanded=True):
            st.markdown(st.session_state.last_analysis)
    if st.session_state.get("last_advices"):
        with st.expander("💡 投资建议", expanded=True):
            st.json(st.session_state.last_advices)
    if not st.session_state.get("last_analysis") and not st.session_state.get("last_advices"):
        st.info("暂无分析报告，请先提问")
    st.divider()
    st.caption("会话 ID: " + st.session_state.config["configurable"]["thread_id"])