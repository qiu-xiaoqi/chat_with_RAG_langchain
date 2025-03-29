import sys
import os
import IPython.display
import io
import gradio as gr
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from make_database.create_db import create_db_info
from chain.chat_QA_chain_self import Chat_QA_chain_self
from chain.QA_chain_self import QA_chain_self
import re


_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k", "deepseek-chat"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "deepseek-chat"
EMBEDDING_MODEL_LIST = ['openai-embedding', 'zhipuai', 'm3e']
INIT_EMBEDDING_MODEL = "openai-embedding"
DEFAULT_DB_PATH = "./knowledge_db"
DEFAULT_PERSIST_PATH = "./vector_db/chroma"

AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "./figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"

BOT_AVATAR_PATH = "./figures/bot_avatar.jpg"
AVATAR_PATH = './figures/avatar.jpg'

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


class Model_center():
    """
    存储问答 Chain 的对象 

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """

    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, 
                                  question: str, 
                                  chat_history: list = [], 
                                  model: str = "deepseek-chat", 
                                  embedding: str = "openai-embedding", 
                                  temperature: float = 0.0, 
                                  top_k: int = 4, 
                                  history_len: int = 3, 
                                  file_path: str = DEFAULT_DB_PATH, 
                                  persist_path: str = DEFAULT_PERSIST_PATH):
        if question == None or len(question) < 1:
            return "", chat_history
        
       
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(model=model, 
                                                                                 temperature=temperature,
                                                                                 top_k=top_k, 
                                                                                 chat_history=chat_history, 
                                                                                 file_path=file_path, 
                                                                                 persist_path=persist_path, 
                                                                                 embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history
        
    def qa_chain_self_answer(self, 
                             question: str, 
                             chat_history: list = [], 
                             model: str = "deepseek", 
                             embedding="openai-embedding", 
                             temperature: float = 0.0, 
                             top_k: int = 4, 
                             file_path: str = DEFAULT_DB_PATH, 
                             persist_path: str = DEFAULT_PERSIST_PATH):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                print(f"{model}, {embedding}")
                self.qa_chain_self[(model, embedding)] = QA_chain_self(model=model, 
                                                                       temperature=temperature,
                                                                       top_k=top_k, 
                                                                       file_path=file_path, 
                                                                       persist_path=persist_path, 
                                                                       embedding=embedding)
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append(
                (question, chain.answer(question=question, temperature=temperature, top_k=top_k))
            )
            return "", chat_history
        except Exception as e:
            return e, chat_history


    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()



def format_chat_prompt(message, chat_history):  
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt


def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    if message == None or len(message) < 1:
        return "", chat_history
    
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history
    
    except Exception as e:
        return e, chat_history
        

def test():
    question = "什么是蘑菇书(easyrl)？"
    model = "deepseek-chat"
    temperature = 0.7
    top_k = 3
    file_path = "../knowledge_db"
    persist_path = r"E:\LLM_Project\RAG\chat_with_RAG_langchain\vector_db"
    model_center = Model_center()
    # 调用 qa_chain_self_answer 方法
    chat_history = []

    response, chat_history = model_center.qa_chain_self_answer(
        question=question,
        model=model,
        temperature=temperature,
        top_k=top_k
    )
    print("Response:", response)
    print("Chat History:", chat_history)


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        # # 中间的图标
        # gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

        # 名字
        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>问答机器人</center></h1>
                        """)
        # # Logo
        # gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False, container=False)

    with gr.Row():
        with gr.Column(scale=4):
            # 聊天记录框
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True, avatar_images=(AVATAR_PATH, BOT_AVATAR_PATH))
            # 输入框
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 三种对话模式的按钮
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            
            with gr.Row():
                # 清除按钮
                clear = gr.ClearButton(components=[chatbot], 
                                       value="Clear console")
        
        with gr.Column(scale=1):
            # 上传知识功能
            file = gr.File(label='请选择知识库目录', file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])
            with gr.Row():
                # 将知识向量化的按钮
                init_db_button = gr.Button("知识库文件向量化")
            
            # 参数配置栏
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(minimum=0, 
                                        maximum=1, 
                                        value=0.1, 
                                        step=0.1,
                                        label="temperature", 
                                        interactive=True)

                top_k = gr.Slider(minimum=1, 
                                  maximum=10, 
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(minimum=0, 
                                        maximum=5,
                                        value=3,
                                        step=1,
                                        label="history len",
                                        interactive=True)

            # 模型选择栏
            model_select = gr.Accordion("模型选择")
            with model_select:
                llm = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True
                )

                embeddings = gr.Dropdown(
                    EMBEDDING_MODEL_LIST,
                    label="Embedding model",
                    value=INIT_EMBEDDING_MODEL,
                    interactive=True
                    )

        ###为每个按钮绑定事件
        # 将知识向量化的按钮绑定事件
        init_db_button.click(
            create_db_info, 
            inputs=[file, embeddings], 
            outputs=[msg], 
            ) 
        
        # 历史数据对话的点击事件
        db_with_his_btn.click(
            model_center.chat_qa_chain_self_answer, 
            inputs=[
                msg,
                chatbot,
                llm,
                embeddings,
                temperature,
                top_k,
                history_len
            ],
            outputs=[msg, chatbot]
        )

        db_wo_his_btn.click(
            model_center.qa_chain_self_answer,
            inputs=[
                msg,
                chatbot,
                llm,
                embeddings,
                temperature,
                top_k,
            ],
            outputs=[msg, chatbot]
        )

        llm_btn.click(
            respond,
            inputs=[
                msg,
                chatbot,
                llm,
                history_len,
                temperature
            ],
            outputs=[msg, chatbot],
            show_progress="minimal"
        )

        # 设置文本框提交事件，默认使用的回答是qa_chain_self_answer
        msg.submit(
            model_center.qa_chain_self_answer,
            inputs=[
                msg,
                chatbot,
                llm,
                embeddings,
                temperature,
                top_k,
            ],
            outputs=[msg, chatbot],
            show_progress="minimal"
        )

        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)

    gr.Markdown("""提醒：<br>
    1. 目前large language model只有deepseek-chat可用
    2. 目前embedding model只有openai-embedding可用
    """)

            
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
# python .\serve\demo.py -model_name='deepseek-chat' -embedding_model='openai-embedding' -db_path='knowledge_db' -persist_path='vector_db'