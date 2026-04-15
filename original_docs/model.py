import os
from langchain_openai import AzureChatOpenAI

def create_gpt_call(temperature: float, max_tokens: int) -> AzureChatOpenAI:
    llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_GPT4O_ENDPOINT"),
                api_key=os.getenv("AZURE_GPT4O_API_KEY"),
                api_version="2025-01-01-preview",
                model="gpt-4o",
                temperature=temperature,
                max_tokens=max_tokens
            )
    if not llm:
        raise ValueError("Azure OpenAI 模型初始化失败，请检查环境变量设置。")
    else:
        print("✅ Azure OpenAI 模型初始化成功！")
        return llm