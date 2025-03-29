from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from chain.model_to_llm import model_to_llm
from chain.get_vectordb import get_vectordb
import re
from dotenv import load_dotenv


class QA_chain_self():
    """"
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型  
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq    
    """

    default_template_rq = """基于以下提供的上下文信息，以专业的客服风格回答用户的问题。
    回答总是以'根据知识库以及你的问题，我向您提供以下答案：'；
    回答的内容应该是事实，而不是猜测；
    回答的内容尽量不分点表示而是完整地表达。
    如果无法找到确切的答案，请礼貌地告知用户您无法提供相关信息，但会尽力协助解决。
    {context}
    用户问题: {question}
    """

    def __init__(self, model:str, temperature:float=0.0, top_k:int=4,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",  embedding_key = None, template=default_template_rq):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                              template=self.template)
        
        self.retriever = self.vectordb.as_retriever(search_type="similarity",
                                                    search_kwargs={'k': self.top_k})
        
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                    retriever=self.retriever,
                                                    return_source_documents=True,
                                                    chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})

    def answer(self, question:str=None, temperature = None, top_k = 4):
        """
        核心方法，调用回答链
        arguments:
        - question: 用户提问
        """

        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature

        if top_k == None:
            top_k = self.top_k

        result = self.qa_chain({"query": question, 
                                "temperature": temperature, 
                                "top_k": top_k})
        result = self.qa_chain({"query": question})
        answer = result["result"]
        answer = re.sub(r"\\n", '<br/>', answer)
        return answer
    

if __name__ == "__main__":
    load_dotenv()
    model = "deepseek-chat"
    temperature = 0.7
    top_k = 3
    file_path = "knowledge_db"
    persist_path = r"D:\code\LLM_Project\RAG\chat_with_RAG_langchain\vector_db\chroma"
    api_key = os.getenv("DeepSeek_API_for_RAG")
    embedding = "openai"
    embedding_key = os.getenv("EMBEDDING_KEY")


    qa_chain = QA_chain_self(
        model=model,
        temperature=temperature,
        top_k=top_k,
        file_path=file_path,
        persist_path=persist_path,
        embedding=embedding,
        embedding_key=embedding_key
    )

    question = "什么是强化学习"

    answer = qa_chain.answer(question=question)

    print("答案:", answer)

    # cd 进入 当前文件所在目录
    # python ./QA_chain_self.py