# refine-rag
refine-rag 是精进 RAG 的意思。
最近浏览一下小红书，经常能刷到后端、前端转 RAG、Agent 工程师的笔记，所以我在学习这些内容之后，也准备分享一下我所学到的的知识（之前学习了黄佳老师的课程），一来是巩固一下自己的知识，二来是分享给有需要的朋友。这个系列教程的大概内容是：

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

因为 LangChain 只支持 Python 和 TS，而我之前就会 Python，为了降低学习成本，所以本系列教程是用 Python 编程语言。

## 快速开始
在根目录新建一个 .env 文件，依次贴入以下密钥即可运行项目代码
```
# DeepSeek API 配置密钥
DEEPSEEK_API_KEY=xxx
# 千问密钥
DASHSCOPE_API_KEY=xxx
# jina 多模态嵌入密钥
JINA_API_KEY=xxx
```
**如何获取 API Key：**
- DeepSeek: https://platform.deepseek.com/
- 千问(DashScope): https://dashscope.aliyun.com/
- JINA_API：https://jina.ai/

### 启动 milvus 向量数据库
部分 Python 文件需要使用 Milvus 向量数据库，按需启动即可
```bash
# 1. 进入目录
cd rag/08-向量存储/

# 2. 启动所有服务
docker compose up -d

# 3. 查看服务状态
docker compose ps

# 4. 查看日志（等待启动完成）
docker compose logs -f milvus-standalone

# 看到这行说明启动成功：
# [INFO] Milvus Proxy successfully started
```

## 公众号
![](https://raw.githubusercontent.com/zonezoen/refine-rag/refs/heads/main/rag/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_2026-03-04_133417_224.jpg)
