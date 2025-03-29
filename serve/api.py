from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)
from chain.QA_chain_self import QA_chain_self

app = FastAPI()


prompt = """
使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”
"""
template = """
{prompt}
{context}
问题：{question}
有用的回答:
"""

class Item(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question : str # 用户 prompt
    model : str = "deepseek-chat"# 使用的模型
    temperature : float = 0.1# 温度系数
    if_history : bool = False # 是否使用历史对话功能
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None
    # APPID
    appid : str = None
    # APISecret
    Spark_api_secret : str = None
    # Secret_key
    Wenxin_secret_key : str = None
    # 数据库路径
    db_path : str = r"D:\code\LLM_Project\RAG\chat_with_RAG_langchain\vector_db\chroma"
    # 源文件路径
    file_path : str = r"D:\code\LLM_Project\RAG\chat_with_RAG_langchain\knowledge_db"
    # prompt template
    prompt_template : str = template
    # Template 变量
    input_variables : list = ["context","question"]
    # Embdding
    embedding : str = "m3e"
    # Top K
    top_k : int = 5
    # embedding_key
    embedding_key : str = None


@app.post("/")
async def get_response(item: Item):

    # 首先确定需要调用的链
    if not item.if_history:
        # 调用 Chat 链
        if item.embedding_key == None:
            item.embedding_key = item.api_key
        chain = QA_chain_self(model=item.model, 
                              temperature=item.temperature, 
                              top_k=item.top_k,
                              file_path=item.file_path,
                              persist_path=item.db_path,
                              api_key=item.api_key,
                              embedding=item.embedding,
                              embedding_key=item.embedding_key,
                              )

        response = chain.answer(question=item.prompt)

        return response

    # 由于 API 存在即时性问题，不能支持历史链
    else:
        return "API 不支持历史链"
    

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

