import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def load_markdown(file_path: str):
    """加载本地 markdown 文件"""
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    print(f"成功加载文档: {file_path}")
    print(f"文档总字符数: {len(docs[0].page_content)}")
    return docs


def split_documents(docs, chunk_size: int = 500, chunk_overlap: int = 50):
    """将文档分割成小块，中文递归切分"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    print(f"\n=== 文档分块统计 ===")
    print(f"分块数量: {len(chunks)}")
    print(f"分块后总字符数: {total_chars}")
    print(f"丢失字符数: {len(docs[0].page_content) - total_chars}")

    return chunks


def init_embeddings(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """初始化中文嵌入模型"""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"嵌入模型初始化完成: {model_name}")
    return embeddings


def build_vectorstore(chunks, embeddings):
    """构建向量存储"""
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)
    print(f"向量存储构建完成，共存储 {len(chunks)} 个文档块")
    return vectorstore


def init_llm(model: str = "deepseek-chat", temperature: float = 0.6):
    """初始化大语言模型"""
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=4096,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    print(f"LLM 初始化完成: {model}")
    return llm


def get_prompt_template():
    """获取提示词模板"""
    prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")
    return prompt


def query_and_answer(vectorstore, llm, prompt, question: str, top_k: int = 3):
    """执行查询并获取答案"""
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print(f"\n=== 检索到 {len(retrieved_docs)} 个相关文档 ===")

    answer = llm.invoke(prompt.format(question=question, context=docs_content))
    return answer, retrieved_docs


def print_token_usage(answer):
    """打印 token 消耗统计"""
    usage = answer.response_metadata.get('token_usage', {})
    print(f"\n=== Token 消耗统计 ===")
    print(f"输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
    print(f"输出 tokens: {usage.get('completion_tokens', 'N/A')}")
    print(f"总 tokens: {usage.get('total_tokens', 'N/A')}")


def main():
    markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

    docs = load_markdown(markdown_path)

    chunks = split_documents(docs)

    embeddings = init_embeddings()

    vectorstore = build_vectorstore(chunks, embeddings)

    llm = init_llm()

    prompt = get_prompt_template()

    question = "文中举了哪些例子？"
    answer, retrieved_docs = query_and_answer(vectorstore, llm, prompt, question)

    print("\n=== AI 回答 ===")
    print(answer.content)

    print_token_usage(answer)


if __name__ == "__main__":
    main()
