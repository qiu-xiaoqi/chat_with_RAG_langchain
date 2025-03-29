import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm.call_llm import parse_llm_api_key
from dotenv import load_dotenv
import os

load_dotenv()
def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    # Huggingface 的 m3e 模型
    # if embedding == 'm3e':
    #     return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    
    if embedding == "openai-embedding":
        return OpenAIEmbeddings(model="text-embedding-ada-002", 
                                openai_api_key=embedding_key, 
                                base_url="https://api.zhizengzeng.com/v1/")
    else: 
        raise ValueError(f"embedding {embedding} not support ")
    

if __name__ == "__main__":
    embedding = "openai"
    embedding_key = os.getenv("EMBEDDING_KEY")
    print(embedding_key)
    embedding_model = get_embedding(embedding, embedding_key)
    print(type(embedding_model))