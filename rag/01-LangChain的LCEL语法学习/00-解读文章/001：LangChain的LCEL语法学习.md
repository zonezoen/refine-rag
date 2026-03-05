> 本文是 [refine-rag](https://github.com/zonezoen/refine-rag) 系列教程的第二篇，带你体验一下LangChain 的 LCEL 语法。
> 本文所有代码都在：https://github.com/zonezoen/refine-rag

## 前言
LCEL (LangChain Expression Language)，语言链表达式语言，是LangChain的表达式语言，用于描述数据处理逻辑。有点类似 Unix 的管道符号。本质就是把前面的输出
传递给后面作为输入。

## 基础使用
`Chain = Prompt | Model | OutputParser`

这个比较简单，就是先写好几个实例，然后用 ｜ 连接起来，形成一个链。想要尽快掌握的话，可以教大模型给你出几道 LCEL 相关的题目给你，写完之后叫AI 给你评分并指导正确答案。

本文所有代码都在：https://github.com/zonezoen/refine-rag
创建 `.env` 文件，放到项目根目录，文件里面写入以下配置信息：
```
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-xxx
# 千问
DASHSCOPE_API_KEY=sk-yyy
```

**如何获取 API Key：**
- DeepSeek: https://platform.deepseek.com/
- 千问(DashScope): https://dashscope.aliyun.com/

```python
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
```
运行结果
```python
周杰伦（Jay Chou），己未年（西元一九七九）正月十八日生于台湾新北。乃华语乐坛巨擘，集歌者、制乐、词曲、导演、伶人于一身，世誉“亚洲流行天王”。

**早年志学**
幼习琴律，天赋初显。丙子年（一九九七）参演《超级新人王》，擢亚魁，其才为吴宗宪所识，遂入阿尔发乐司为助。初为他人撰曲（若《眼泪知道》《双截棍》皆尝见拒），曲风新奇，时人未尽纳。

**乐业腾骞**
*   **首辑破局**：庚辰年（二〇〇〇）发专辑《Jay》，融R&B、嘻哈并中国雅韵，一举革新乐坛旧制，夺金曲奖最佳专辑。
*   **开宗中国风**：以传统文脉入流行宫商，作《东风破》《发如雪》《青花瓷》《千里之外》诸名篇，倡“中国风”为一时潮流。
*   **丰产屡进**：嗣后岁出一辑，若《范特西》《叶惠美》《七里香》《十一月的萧邦》等，皆风行四海。曲格多元，兼摄古典、摇滚、电子；其词多与方文山共谋，诗象盎然。
*   **荣冠载途**：累获金曲奖十五座，销录屡破，实千禧年后华语乐坛之圭臬。

**跨界骋才**
*   **影戏**：乙酉年（二〇〇五）初主演《头文字D》，获金马奖新人魁首。丁亥年（二〇〇七）自导自演《不能说的秘密》，成乐影合璧之经典。
*   **导艺与演游**：掌镜多部曲影，戊戌年（二〇一八）主创《周游记》。壬寅年（二〇二二）监制并出演好莱坞影戏《极限追杀》。
*   **商略**：创潮服牌“PHANTACi”、电竞赛旅“J Team”，亦涉饮馔诸业。
```

## RunnableParallel和RunnablePassthrough
现实中，数据流不会总是这么简单。有时候你需要同时处理多个输入，或者把某些数据原样传递。这时候就需要 **`RunnableParallel`** 和 **`RunnablePassthrough`**。

### 2.1 RunnableParallel (并行处理)

当你需要把一份输入分发给多个组件，或者构造一个复杂的字典传给下一个环节时，用它。

### 2.2 RunnablePassthrough (原样通过)

它像是一个透明的管道，把输入数据直接传给下一级，不做任何修改。

代码示例：
```python
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
```

ok，这些你掌握之后，就算基本掌握了 LCEL 的语法了，后续学习中可能会陆陆续续用到这些知识。

## 学习路径

1. 简易RAG 学习
2. LCEL 语法学习
3. LangChain 读取数据
   1. LangChain 读取文本数据
   2. LangChain 读取图片数据
   3. LangChain 读取 PDF 数据
   4. LangChain 读取表格数据
4. 文本切块
5. 向量嵌入
6. 向量存储
7. 检索前处理
8. 索引优化
9. 检索后处理
10. 响应生成
11. 系统评估

## 项目地址

本文所有代码示例都在 GitHub 开源：

https://github.com/zonezoen/refine-rag

欢迎 Star 和 Fork，一起学习 RAG 技术！