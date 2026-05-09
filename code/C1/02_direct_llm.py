import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.6,
    max_tokens=4096,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 测试问题
question = "1+1等于几？"
context = "数学中，1加1等于2。"

# 调用 LLM
answer = llm.invoke(prompt.format(question=question, context=context))
print(answer.content)

# Token 消耗统计
usage = answer.response_metadata.get('token_usage', {})
print(f"\n=== Token 消耗统计 ===")
print(f"输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
print(f"输出 tokens: {usage.get('completion_tokens', 'N/A')}")
print(f"总 tokens: {usage.get('total_tokens', 'N/A')}")
