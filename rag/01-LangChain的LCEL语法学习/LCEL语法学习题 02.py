from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

prompt_01 = ChatPromptTemplate.from_template(
    """
    生成该公司的“口号（Slogan）：{company}
    """
)

prompt_02 = ChatPromptTemplate.from_template(
    """
    预测该公司“明年的股价”：{company}
    """
)

llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=0.7,  # 随机性
    max_tokens=2048,  # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)

chain1 = {"company": RunnablePassthrough()} | prompt_01 | llm | StrOutputParser()
chain2 = {"company": RunnablePassthrough()} | prompt_02 | llm | StrOutputParser()
chain = RunnableParallel({"slogan": chain1, "stock": chain2})

answer = chain.invoke("百度")
print(answer)
