import os
import pandas as pd
from .metadata import load_info  #.metadata表示相对路径，表示从当前文件所在目录开始寻找

def load_episode_parquet(dataset_path: str, episode_idx: int) -> pd.DataFrame:
    """
    读取指定episode的parquet文件，返回DataFrame
    根据episode_idx自动计算chunk编号，按命名规则拼出文件路径
    """
    chunk = episode_idx // 1000

    chunk_path = f"chunk-{chunk:03d}"
    episode_path = f"episode_{episode_idx:06d}.parquet"
    path = os.path.join(dataset_path, "data", chunk_path, episode_path)

    episode = pd.read_parquet(path)
    return episode

def get_video_path(dataset_path: str, episode_idx: int, camera_name: str) -> str:
    """
    根据info.json中的video_path模板，拼出某个episode某路相机的视频文件路径。
    模板示例："videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    """
    info = load_info(dataset_path)
    video_path = info["video_path"]
    chunk_idx = episode_idx // info["chunks_size"]

    relative_path = video_path.format(episode_chunk=chunk_idx, video_key=camera_name, episode_index=episode_idx)
    path = os.path.join(dataset_path, relative_path)
    return path


