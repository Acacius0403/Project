import json
import os

def load_info(dataset_path: str) -> dict:
    "读取 meta/info.json 文件，返回数据集元信息字典"
    json_path = os.path.join(dataset_path, "meta", "info.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_tasks(dataset_path: str) -> list[dict]:
    "读取 meta/tasks.jsonl 文件，每行一个任务，返回任务字典组成的列表"
    jsonl_path = os.path.join(dataset_path, "meta", "tasks.jsonl")
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            task = json.loads(line)
            tasks.append(task)
    return tasks

def load_episodes(dataset_path: str) -> list[dict]:
    "读取 meta/episodes.jsonl 文件，每行一个 episode，返回 episode 字典组成的列表"
    jsonl_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            episode = json.loads(line)
            episodes.append(episode)
    return episodes



