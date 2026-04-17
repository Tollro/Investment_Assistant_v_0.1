from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from test.model import create_gpt_call
from test.stock_list_db_tools import query_by_name_keyword as _query_by_name_keyword
from test.akshare_tools import get_stock_daily_data as _get_stock_daily_data
from test.akshare_tools import get_financial_report as _get_financial_report
from test.akshare_tools import get_news_titles as _get_news_titles
from typing import Literal
from pydantic import BaseModel, Field

class GetStockKlineDataInput(BaseModel):
    symbol: str = Field(description="股票代码（必须包含前缀），如 'sz000001'")
    start_date: str = Field(description="开始日期，格式 YYYYMMDD")
    end_date: str = Field(description="结束日期，格式 YYYYMMDD")
    adjust: Literal["", "hfq", "qfq"] = Field(
        description="复权类型：空字符串''表示不复权，'hfq' 表示后复权，'qfq' 表示前复权"
    )

class GetNewsTitlesInput(BaseModel):
    symbol: str = Field(description="股票代码，如 'sz000001'")
    top_n: int = Field(description="返回当日最近top_n条")

class GetFinancialReportInput(BaseModel):
    symbol: str = Field(description="股票代码，如 'sz000001'")
    report_type: str = Field(description="报表类型，可选 '资产负债表', '利润表', '现金流量表'其中之一")


@tool
def get_stock_code(name:str) -> list[str]:
    """根据股票名称查找股票代码"""
    return _query_by_name_keyword(keyword=name)

@tool(args_schema=GetStockKlineDataInput)
def get_stock_kline_data(symbol: str, start_date: str, end_date: str, adjust:Literal["", "hfq", "qfq"]):
    """根据股票代码、日期范围获取股票日线数据"""
    return _get_stock_daily_data(symbol, start_date, end_date)

@tool(args_schema=GetNewsTitlesInput)
def get_news_titles(symbol, top_n=20):
    """根据股票代码获取新闻标题"""
    return _get_news_titles(symbol, top_n)

@tool(args_schema=GetFinancialReportInput)
def get_financial_report(symbol, report_type=Literal["资产负债表", "利润表", "现金流量表"]):
    """根据股票代码获取财务报表"""
    return _get_financial_report(symbol, report_type)

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

class ResearcherState(TypedDict):
    # add_messages 是一个内置的 reducer，新消息被追加，而不是覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    structured_analysis: Optional[str]


if __name__ == "__main__":

    memory = MemorySaver()
    tools = [get_stock_code, get_stock_kline_data, get_news_titles, get_financial_report]

    llm = create_gpt_call(temperature=0.5, max_tokens=4096)

    graph = create_agent(
        model=llm, 
        tools=tools,
        system_prompt="""
            你是一个资深的的财经信息整理助手，拥有多年的研究经验。你必须根据用户意图选择工具，可以同时调用多个工具来完善答案：
            1. **查询新闻/资讯/公告/舆情**：
            - 工具：get_news_titles
            2. **查询股价走势/技术分析/K线图**：
            - 工具：get_stock_kline_data
            3. **查询财务报表/基本面/业绩**：
            - 工具：get_financial_report
            4. **根据股票/企业/公司的名称获取股票代码**:
            - 工具：get_stock_code
            注意：
            - 若用户希望查询相关新闻，请对内容进行提炼和总结。若多条新闻标题完全一样，则只需要提及一次。
            每条资讯可按照下面的格式进行提炼和总结
            '
            ## 一、核心信息
            - **事件概要**：用1-2句话精准概括发生了什么
            - **涉及主体**：列出相关公司/行业/政策/人物
            - **关键数据**：提取核心数字（股价、金额、增长率、估值等）

            ## 二、多维度分析
            1. **事件性质**
            - 属于哪类事件？（政策/业绩/并购/行业变动等）
            - 是突发性还是延续性？

            2. **影响评估**
            - 对涉事主体的影响（短期/中长期）
            - 对所属行业的连锁反应
            - 对整体市场的信号意义

            3. **利好/利空拆解**
            - 利好因素有哪些？
            - 利空因素有哪些？
            - 哪些被市场过度解读/低估？

            4. **风险提示**
            - 潜在的不确定性
            - 需要警惕的信号

            ## 三、投资参考
            - 对不同类型投资者的启示（短线/中线/长线）
            - 需要跟踪的后续信息
            - 建议观察的时间窗口

            ## 四、一句话总结
            用一句话说出这件事最核心的价值/风险点。

            ---
            要求：
            - 观点要有依据，不臆测
            - 数据要精准，不错漏
            - 保持客观中立，但可以有独立判断
            - 语言简洁，避免冗余

            新闻内容：
            [粘贴新闻]
            '
            - 若用户输入股票/企业/公司的名称，先调用get_stock_code获取股票代码。
        """,
        middleware=[handle_tool_errors], 
        state_schema=ResearcherState,            
        checkpointer=memory
    )
    
    session_config = {"configurable": {"thread_id": "user_001"}}

    result = graph.invoke(
        {"messages": [("user", "告诉我国盾的情况")]},
        config=session_config
    )
    print("\n" + "="*50)
    print("Agent 执行流程")
    print("="*50)
    print_messages_simple(result["messages"])
