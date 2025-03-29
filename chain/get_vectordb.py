import sys

from langchain_openai import OpenAIEmbeddings
import os
from make_database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding


def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "openai",embedding_key:str=None):
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            vectordb = create_db(file_path, persist_path, embedding)
            vectordb = load_knowledge_db(persist_path, embedding)

        else:
            vectordb = load_knowledge_db(persist_path, embedding)
    else:
        vectordb = create_db(file_path, persist_path, embedding)
        vectordb = load_knowledge_db(persist_path, embedding)
    
    return vectordb




