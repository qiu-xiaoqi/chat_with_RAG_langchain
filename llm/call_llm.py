from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlparse
import os
from datetime import datetime
from langchain.utils import get_from_dict_or_env

def parse_llm_api_key(model:str, env_file:dict()=None):
    """
    通过 model 和 env_file 的来解析平台参数
    """   
    # 加载API KEY
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai-embedding":
        return env_file["EMBEDDING_KEY"]
    
    if model == "deepseek":
        return env_file["DeepSeek_API_for_RAG"]

    else:
        raise ValueError(f"model{model} not support!!!")
    

def get_completion(prompt :str, model :str, temperature=0.1,api_key=None, secret_key=None, access_token=None, appid=None, api_secret=None, max_tokens=2048):
    # 调用大模型获取回复，支持上述三种模型+gpt
    # arguments:
    # prompt: 输入提示
    # model：模型名
    # temperature: 温度系数
    # api_key：如名
    # secret_key, access_token：调用文心系列模型需要
    # appid, api_secret: 调用星火系列模型需要
    # max_tokens : 返回最长序列
    # return: 模型返回，字符串
    # 调用 GPT
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return "暂时还没有这个模型"
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        return "暂时还没有这个模型"
    elif model in ["Spark-1.5", "Spark-2.0"]:
        return "暂时还没有这个模型"
    elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
        return "暂时还没有这个模型"
    
    elif model in ["deepseek-chat"]:
        return get_completion_deepseek(prompt, model, temperature, api_key, max_tokens)
    else:
        return "不正确的模型"
    
def get_completion_deepseek(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int, base_url="https://api.deepseek.com"):
    # 封装 OpenAI 原生接口
    if api_key == None:
        api_key = parse_llm_api_key("deepseek")

    client = OpenAI(api_key=api_key, base_url=base_url)
    # 具体调用
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # 模型输出的温度系数，控制输出的随机程度
        max_tokens = max_tokens, # 回复最大长度
    )
    
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message.content

if __name__ == "__main__":
    text = "你好，请问你是什么模型？"
    model = "deepseek-chat"
    temperature = 0.1
    max_tokens = 2048
    base_url = "https://api.deepseek.com"
    llm = get_completion_deepseek(text, model, temperature, None, max_tokens, base_url)
    print(llm)
