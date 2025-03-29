import json
import requests
import os
import loguru
from dotenv import load_dotenv
import base64
import shutil
# 加载Github的TOKEN

load_dotenv()
TOKEN = os.getenv('GITHUB_TOKEN')
if TOKEN is None:
    print("--------------- Github Token is not set -------------------")
else:
    print("--------------- Github Token is set -------------------")


def get_repos(org_name, token, export_dir):
    headers = {
        'Authorization': f'token {token}'
    }
    url = f'https://api.github.com/orgs/{org_name}/repos'
    response = requests.get(url, headers=headers, params={'per_page': 200, 'page': 0})
    if response.status_code == 200:
        # 返回所有仓库的信息,并将所有仓库名保存到repositories.txt中
        repos = response.json()
        loguru.logger.info(f"Fetched **{len(repos)}** repositories for **{org_name}**")
        repositories_path = os.path.join(export_dir, 'repositories.txt')
        with open(repositories_path, 'w', encoding='utf-8') as file:
            for repo in repos:
                file.write(repo['name'] + '\n')
        return repos

    else:
        loguru.logger.error(f"Error fetching repositories: {response.status_code}")
        loguru.logger.error(response.text)
        return []

def fetch_repo_readme(org_name, repo_name, token, export_dir):
    headers = {
        'Authorization': f'token {token}'
    }
    url = f'https://api.github.com/repos/{org_name}/{repo_name}/readme'
    print(f"Fetching README for {repo_name}...")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(f"Fetched README for {repo_name}")
        readme_content = response.json()['content']
        # 将 Base64 编码的 README 内容解码并保存到文件中
        readme_content = base64.b64decode(readme_content).decode('utf-8')
        # 将README内容保存到文件中
        repo_dir = os.path.join(export_dir, repo_name)
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)
        readme_path = os.path.join(repo_dir, 'README.md')
        # 文件不为空则跳过写入
        if os.path.exists(readme_path) and os.path.getsize(readme_path) > 0:
            print(f"README.md for {repo_name} already exists and is not empty. Skipping write.")
        else:
            with open(readme_path, 'w', encoding='utf-8') as file:
                file.write(readme_content)
    else:
        loguru.logger.error(f"Error fetching README for {repo_name}: {response.status_code}")
        loguru.logger.error(response.text)
        
if __name__ == '__main__':
    # datawhale的路径：https://github.com/datawhalechina
    org_name = 'datawhalechina'
    export_dir = "make_database/readme_db"
    repos  = get_repos(org_name, TOKEN, export_dir)

    if repos:
        for repo in repos:
            repe_name = repo['name']
            fetch_repo_readme(org_name, repe_name, TOKEN, export_dir)

    if os.path.exists('temp'):
        shutil.rmtree('temp')