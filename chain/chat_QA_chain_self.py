from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import sys
sys.path.append("../")
from chain.model_to_llm import model_to_llm
from chain.get_vectordb import get_vectordb
import re


class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """

    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",embedding_key:str=None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        
        self.default_template_rq = """基于以下提供的上下文信息，以专业的客服风格回答用户的问题。
        回答总是以'根据知识库以及你的问题，我向您提供以下答案：'；
        回答的内容应该是事实，而不是猜测；
        回答的内容尽量不分点表示而是完整地表达。
        如果无法找到确切的答案，请礼貌地告知用户您无法提供相关信息，但会尽力协助解决。
        {context}
        用户问题: {question}
        """

        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()
    
    def change_history_length(self,history_len:int=1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]
    
    def answer(self, question:str=None,temperature = None, top_k = 4):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """

        if len(question) == 0:
            return "", self.chat_history
        
        if temperature == None:
            temperature = self.temperature
        
        llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret, self.Wenxin_secret_key)

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = self.vectordb.as_retriever(search_type="similarity",   
                                               search_kwargs={'k': top_k})


        qa_chain_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.default_template_rq
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,
            memory = self.memory,
            combine_docs_chain_kwargs={"prompt": qa_chain_prompt}
        )

        result = qa({"question": question, 
                     "chat_history": self.chat_history})
        
        answer = result["answer"]
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question, answer))

        return self.chat_history