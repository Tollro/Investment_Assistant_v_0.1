from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from original_docs.model import create_gpt_call
from test.stock_list_db_tools import query_by_name_keyword as _query_by_name_keyword
from test.akshare_tools import get_stock_daily_data as _get_stock_daily_data
from test.akshare_tools import get_financial_report as _get_financial_report
from test.akshare_tools import get_news_titles as _get_news_titles
from typing import Literal
from pydantic import BaseModel, Field

from test.graph import InvestmentState

# class GetStockKlineDataInput(BaseModel):
#     symbol: str = Field(description="股票代码（必须包含前缀），如 'sz000001'")
#     start_date: str = Field(description="开始日期，格式 YYYYMMDD")
#     end_date: str = Field(description="结束日期，格式 YYYYMMDD")
#     adjust: Literal["", "hfq", "qfq"] = Field(
#         description="复权类型：空字符串''表示不复权，'hfq' 表示后复权，'qfq' 表示前复权"
#     )

# class GetNewsTitlesInput(BaseModel):
#     symbol: str = Field(description="股票代码，如 'sz000001'")
#     top_n: int = Field(description="返回当日最近top_n条")

# class GetFinancialReportInput(BaseModel):
#     symbol: str = Field(description="股票代码，如 'sz000001'")
#     report_type: str = Field(description="报表类型，可选 '资产负债表', '利润表', '现金流量表'其中之一")


# @tool
# def get_stock_code(name:str) -> list[str]:
#     """根据股票名称查找股票代码"""
#     return _query_by_name_keyword(keyword=name)

# @tool(args_schema=GetStockKlineDataInput)
# def get_stock_kline_data(symbol: str, start_date: str, end_date: str, adjust:Literal["", "hfq", "qfq"]):
#     """根据股票代码、日期范围获取股票日线数据"""
#     return _get_stock_daily_data(symbol, start_date, end_date)

# @tool(args_schema=GetNewsTitlesInput)
# def get_news_titles(symbol, top_n=20):
#     """根据股票代码获取新闻标题"""
#     return _get_news_titles(symbol, top_n)

# @tool(args_schema=GetFinancialReportInput)
# def get_financial_report(symbol, report_type=Literal["资产负债表", "利润表", "现金流量表"]):
#     """根据股票代码获取财务报表"""
#     return _get_financial_report(symbol, report_type)

# @wrap_tool_call
# def handle_tool_errors(request, handler):
#     """使用自定义消息处理工具执行错误。"""
#     try:
#         return handler(request)
#     except Exception as e:
#         # 向模型返回自定义错误消息
#         return ToolMessage(
#             content=f"工具错误：请检查您的输入并重试。({str(e)})",
#             tool_call_id=request.tool_call["id"]
#         )
    
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


def researcher_node(state: InvestmentState) -> dict:
    """
    本节点负责：
    1. 从用户消息中提取意图（股票名称或代码）
    2. 调用工具收集财务数据、新闻标题
    3. 将数据整理为结构化字典存入 state
    """
    user_input = state["messages"][-1].content
    collected = {}

    # 1. 解析股票名称 -> 代码（简单示意，可用 LLM 提取实体）
    # 这里假设用户输入包含公司名称，调用 get_stock_code

    # 此处查询应放在chatbot节点。根据查询结果数量处理
    stock_name = "国盾"  # 实际应用中可用正则或 LLM 抽取
    stock_codes = _query_by_name_keyword(keyword=stock_name)
    if not stock_codes:
        raise ValueError("未找到对应股票")
    symbol = stock_codes[0]  # 取第一个结果
    collected["stock_code"] = symbol
    collected["company_name"] = stock_name  # 实际可从数据库获取全称
    

    # 2. 获取三张财务报表（可异步并发）
    collected["financial_reports"] = {}
    for report_type in ["资产负债表", "利润表", "现金流量表"]:
        try:
            report_data = _get_financial_report(symbol, report_type)
            collected["financial_reports"][report_type] = report_data
        except Exception as e:
            collected["financial_reports"][report_type] = {"error": str(e)}

    # 3. 获取新闻标题
    news_data = _get_news_titles(symbol, top_n=20)
    collected["news_titles"] = news_data

    # 4. 可选：获取 K 线数据（如果需要）
    # kline = _get_stock_daily_data(symbol, "20250101", "20260101")
    # collected["kline_data"] = kline

    return {
        "collected_data": collected,
        # 也可以把工具调用过程的消息记录到 messages 中，便于调试
        "messages": [AIMessage(content=f"已收集 {symbol} 的数据")]
    }
