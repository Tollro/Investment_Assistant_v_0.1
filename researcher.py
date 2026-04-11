from langchain.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from model import create_gpt_call
from akshare_tools.stock_list_db_tools import query_by_name_keyword as _query_by_name_keyword
from akshare_tools.akshare_tools import get_stock_daily_data as _get_stock_daily_data

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取一个消息历史记录对象，支持会话 ID 来区分不同用户的历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@tool
def get_stock_code(name:str) -> list[str]:
    """根据股票名称查找股票代码"""
    return _query_by_name_keyword(keyword=name)

@tool
def get_stock_daily_data(symbol: str, start_date: str, end_date: str):
    """根据股票代码和日期范围获取股票日线数据"""
    return _get_stock_daily_data(symbol, start_date, end_date)

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

if __name__ == "__main__":

    memory = MemorySaver()
    tools = [get_stock_code, get_stock_daily_data]

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "你是一个投资研究助手，帮助用户查询股票信息。用户可能输入股票名称来查询代码，或提供股票代码和时间范围来查询数据。请根据用户意图使用相应工具。"),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{input}")
    #     ]
    # )

    llm = create_gpt_call(temperature=0.2, max_tokens=800)

    graph = create_agent(
        model=llm, 
        tools=tools,
        system_prompt="你是一个投资研究助手，帮助用户查询股票信息。用户可能输入股票名称来查询代码，或提供股票代码和时间范围来查询数据。请根据用户意图使用相应工具。若已有的数据足够用来回答该问题则直接回复用户。",
        middleware=[handle_tool_errors],             
        checkpointer=memory
    )

    result = graph.invoke(
        {"messages": [("user", "查询平安银行的股票信息，从2023-01-01到2023-2-01")]}
    )
    print("\n" + "="*50)
    print("Agent 执行流程")
    print("="*50)
    print_messages_simple(result["messages"])
