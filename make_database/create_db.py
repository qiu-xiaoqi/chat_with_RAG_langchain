import os
import sys
import re
# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# 打印当前路径
print(os.getcwd())
import tempfile
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
load_dotenv()
DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db"

def get_files(dir_path):
    file_list = []
    # 遍历目录下的所有文件
    for filepath, dirnames, filenames in os.walk(dir_path): 
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders):
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
        return 

    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        pattern = r"不存在|风控"
        match = re.search(pattern, file)
        if not match:
            loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return 


def create_db_info(files=DEFAULT_DB_PATH, embeddings="openai-embedding", persist_directory=DEFAULT_PERSIST_PATH):
    if embeddings == 'openai-embedding':
        vectordb = create_db(files, persist_directory, embeddings)
        print("数据向量化完成，现在你可以进行提问了！！！")
    return ""

def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="openai-embedding"):
    """
    该函数用于加载 PDF 文件，切分文档，生成文档的嵌入向量，创建向量数据库。

    参数:
    file: 存放文件的路径。
    embeddings: 用于生产 Embedding 的模型

    返回:
    vectordb: 创建的数据库。
    """
    if files == None:
        return "can't load empty file"
    
    if type(files) != list:
        files = [files]
    
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=150
    )

    split_docs = text_splitter.split_documents(docs)

    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings, embedding_key=os.getenv("embedding_key"))

    persist_directory = './vector_db/chroma'
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()
    return vectordb


def persist_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb

if __name__ == "__main__":
    create_db(embeddings="openai-embedding")