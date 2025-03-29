import sys
sys.path.append("../")
from llm.call_llm import parse_llm_api_key
from langchain.chat_models import ChatOpenAI


def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None,Spark_api_secret:str=None,Wenxin_secret_key:str=None):
    """
    OpenAI & DeepSeek: model,temperature,api_key
    """
    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k", "deepseek-chat"]:
        if api_key == None:
            api_key = parse_llm_api_key("deepseek")
        llm = ChatOpenAI(model_name = model, 
                         temperature = temperature , 
                         openai_api_key = api_key,
                         base_url="https://api.deepseek.com")
    
    return llm
        
