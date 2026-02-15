from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt_01 = ChatPromptTemplate.from_template(
    """
    介绍这个人物的生平事迹：{person}
    """
)

prompt_02 = ChatPromptTemplate.from_template(
    """
    把这个介绍转换成文言文：{text}
    """
)

llm = ChatDeepSeek(
    model="deepseek-chat",  # DeepSeek API 支持的模型名称
    temperature=0.7,  # 随机性
    max_tokens=2048,  # 最大输出长度
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量加载API key
)
ps = {"text": (prompt_01 | llm | StrOutputParser())}
chain = ps | prompt_02 | llm | StrOutputParser()

answer = chain.invoke({"person": "周杰伦"})
print(answer)
