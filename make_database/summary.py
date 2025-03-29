import os 
from dotenv import load_dotenv
from openai import OpenAI 
from get_repo import get_repos
from bs4 import BeautifulSoup 
import markdown
import re
import time
import openai


load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
api_key = os.getenv("DeepSeek_API_for_RAG")
base_url = "https://api.deepseek.com"

def remove_urls(text):
    """
    过滤文本中链接防止大语言模型风控
    """

    # 匹配ulr并替换为空字符串
    url_pattern = re.compile(r'https?://[^\s]*')
    text = re.sub(url_pattern, '', text)

    # 匹配特定的文本并替换为空字符串
    specific_text_pattern = re.compile(r'扫描下方二维码关注公众号|提取码|关注|科学上网|回复关键词|侵权|版权|致谢|引用|LICENSE'
                                       r'|组队打卡|任务打卡|组队学习的那些事|学习周期|开源内容|打卡|组队学习|链接')

    text = re.sub(specific_text_pattern, '', text)
    return text


def extract_text_from_md(md_content):
    """
    抽取md中的文本
    """

    # 将md转换为html
    html = markdown.markdown(md_content)
    # 使用BeautifulSoup提取文本
    soup = BeautifulSoup(html, 'html.parser')

    return remove_urls(soup.get_text())


def generate_llm_summary(repo_name, readme_content, model):
    prompt = f"1：这个仓库名是 {repo_name}. 此仓库的readme全部内容是: {readme_content}\
               2:请用约200以内的中文概括这个仓库readme的内容,返回的概括格式要求：这个仓库名是...,这仓库内容主要是..."
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    messages = [{"role": "system", "content": "你是专业的文本摘要助手"},
               {"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def main(org_name,export_dir,summary_dir,model):
    repos = get_repos(org_name, TOKEN, export_dir)

    os.makedirs(summary_dir, exist_ok=True)
    for id, repo in enumerate(repos):
        repo_name = repo['name']
        readme_path = os.path.join(export_dir, repo_name, 'README.md')
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
            
            readme_text = extract_text_from_md(readme_content)

            time.sleep(60)
            print('第' + str(id) + '条' + f'开始summary{repo_name}')
            try:
                summary = generate_llm_summary(repo_name, readme_text, model)
                print(f"{repo_name}的摘要:" + summary)
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary\n\n")
                    summary_file.write(summary)
            except openai.OpenAIError as e:
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary风控.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary风控\n\n")
                    summary_file.write("README内容风控。\n")
                print(f"Error generating summary for {repo_name}: {e}")

        else:
            print(f"文件不存在: {readme_path}")
            summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary不存在.md")
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(f"# {repo_name} Summary不存在\n\n")
                summary_file.write("README文件不存在。\n")


if __name__ == '__main__':
    org_name = 'datawhalechina'
    export_dir = 'make_database/readme_db'
    summary_dir = 'knowledge_db/readme_summary'
    model = 'deepseek-chat'
    main(org_name, export_dir, summary_dir, model)
