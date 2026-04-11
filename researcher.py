from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from model import create_gpt_call
from akshare_tools.stock_list_db_tools import query_by_name_keyword as _query_by_name_keyword
from akshare_tools.akshare_tools import get_stock_daily_data as _get_stock_daily_data
from typing import Literal
from pydantic import BaseModel, Field

class GetStockDataInput(BaseModel):
    symbol: str = Field(description="股票代码（必须包含前缀），如 'sz000001'")
    start_date: str = Field(description="开始日期，格式 YYYYMMDD")
    end_date: str = Field(description="结束日期，格式 YYYYMMDD")
    adjust: Literal["", "hfq", "qfq"] = Field(
        description="复权类型：空字符串''表示不复权，'hfq' 表示后复权，'qfq' 表示前复权"
    )

@tool
def get_stock_code(name:str) -> list[str]:
    """根据股票名称查找股票代码"""
    return _query_by_name_keyword(keyword=name)

@tool(args_schema=GetStockDataInput)
def get_stock_daily_data(symbol: str, start_date: str, end_date: str, adjust:Literal["", "hfq", "qfq"]):
    """根据股票代码、日期范围获取股票日线数据"""
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

class AgentState(TypedDict):
    # add_messages 是一个内置的 reducer，新消息被追加，而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]

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
        system_prompt="""
                你是一个投资研究助手，帮助用户查询股票信息。
                用户可能输入股票名称或股票代码来查询股票信息。
                请根据用户意图使用相应工具。若已有的数据足够用来回答该问题则直接回复用户。
            """,
        middleware=[handle_tool_errors], 
        state_schema=AgentState,            
        checkpointer=memory
    )
    
    session_config = {"configurable": {"thread_id": "user_001"}}

    result = graph.invoke(
        {"messages": [("user", "查询600012的股票信息，从2023-01-01到2023-2-01")]},
        config=session_config
    )
    print("\n" + "="*50)
    print("Agent 执行流程")
    print("="*50)
    print_messages_simple(result["messages"])
