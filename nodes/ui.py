import streamlit as st
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
import sys
from pathlib import Path

# 添加项目根目录到系统路径，以便导入父图
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入您已编译好的图构建函数
from nodes.Stock_Graph_Single import Stock_Graph_Single  # 请替换为实际文件名

# ---------- 页面配置 ----------
st.set_page_config(page_title="智能投资助手", page_icon="📈", layout="wide")
st.title("📈 智能投资助手")
st.caption("基于 LangGraph 的多智能体协作系统 —— 支持股票查询、分析与建议")

# ---------- 初始化会话状态 ----------
if "assistant" not in st.session_state:
    st.session_state.assistant = Stock_Graph_Single()

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "ui_session_001"}}

if "messages" not in st.session_state:
    # 对话历史展示用
    st.session_state.messages = []

if "interrupt_event" not in st.session_state:
    # 存储中断信息，格式：{"prompt": str, "resume_needed": bool}
    st.session_state.interrupt_event = None

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ---------- 辅助函数 ----------
def run_graph_until_interrupt_or_end(user_input: str = None):
    """
    运行图，直到遇到中断或结束。
    如果 user_input 为 None，表示开始新查询；
    否则表示用 Command(resume=user_input) 恢复执行。
    """
    assistant = st.session_state.assistant
    config = st.session_state.config

    # 构建输入状态
    if user_input is None:
        # 开始新查询
        input_state = {
            "user_query": st.session_state.pending_input,
            "conversation_history": [HumanMessage(content=st.session_state.pending_input)]
        }
    else:
        # 恢复中断
        input_state = Command(resume=user_input)

    events = assistant.stream(input_state, config)
    final_response = None

    for event in events:
        if "__interrupt__" in event:
            # 遇到中断，保存中断信息并暂停
            interrupt_info = event["__interrupt__"][0]
            st.session_state.interrupt_event = {
                "prompt": interrupt_info.value["question"],
                "resume_needed": True
            }
            # 将中断提示作为助手消息添加到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❓ {interrupt_info.value['question']}"
            })
            return None  # 暂停执行

        # 处理正常节点输出
        for node_name, node_output in event.items():
            if isinstance(node_output, dict) and "final_response" in node_output:
                final_response = node_output["final_response"]

    # 执行完毕，清除中断状态
    st.session_state.interrupt_event = None
    return final_response


def process_user_query(prompt: str):
    """用户提交新查询时的处理入口"""
    # 将用户消息加入展示历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_input = prompt

    # 执行图
    final_response = run_graph_until_interrupt_or_end(user_input=None)

    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})

    # 清空临时输入
    st.session_state.pending_input = None

    # 从图状态中获取更多结构化信息（如分析报告、建议）
    final_state = st.session_state.assistant.get_state(st.session_state.config).values
    if final_state:
        # 可以在此处将分析、建议存入 session_state 供额外展示（非必需）
        if final_state.get("analysis"):
            st.session_state.last_analysis = final_state["analysis"]
        if final_state.get("advices"):
            st.session_state.last_advices = final_state["advices"]
        if final_state.get("stock_name"):
            st.session_state.last_stock_name = final_state["stock_name"]


def resume_with_user_answer(answer: str):
    """用户回答中断问题后的处理入口"""
    # 将用户回答加入展示历史
    st.session_state.messages.append({"role": "user", "content": answer})

    # 恢复执行
    final_response = run_graph_until_interrupt_or_end(user_input=answer)

    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})

    # 清除中断状态
    st.session_state.interrupt_event = None

    # 获取最新状态（可选）
    final_state = st.session_state.assistant.get_state(st.session_state.config).values
    if final_state:
        if final_state.get("analysis"):
            st.session_state.last_analysis = final_state["analysis"]
        if final_state.get("advices"):
            st.session_state.last_advices = final_state["advices"]


# ---------- 渲染对话历史 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 处理用户输入 ----------
# 如果有待处理的中断，显示特殊提示
if st.session_state.interrupt_event is not None:
    prompt_placeholder = st.session_state.interrupt_event["prompt"]
    with st.chat_message("assistant"):
        st.info(f"⏸️ 等待您的回答：{prompt_placeholder}")
else:
    prompt_placeholder = "请输入您的问题，例如：分析一下贵州茅台"

# 输入框
user_input = st.chat_input(prompt_placeholder)

if user_input:
    if st.session_state.interrupt_event is not None:
        # 当前处于中断等待状态，用户输入作为回答
        resume_with_user_answer(user_input)
    else:
        # 新的一轮查询
        process_user_query(user_input)

    # 重新运行脚本以刷新界面
    st.rerun()

# ---------- 侧边栏：展示详细分析结果（可选） ----------
with st.sidebar:
    st.header("📋 最新分析报告")
    if "last_stock_name" in st.session_state and st.session_state.last_stock_name:
        st.subheader(f"📌 {st.session_state.last_stock_name}")
    if "last_analysis" in st.session_state and st.session_state.last_analysis:
        with st.expander("🔍 详细分析", expanded=True):
            st.markdown(st.session_state.last_analysis)
    if "last_advices" in st.session_state and st.session_state.last_advices:
        with st.expander("💡 投资建议", expanded=True):
            st.json(st.session_state.last_advices)
    if not st.session_state.get("last_analysis") and not st.session_state.get("last_advices"):
        st.info("暂无分析报告，请先提问")

    st.divider()
    st.caption("会话 ID: " + st.session_state.config["configurable"]["thread_id"])