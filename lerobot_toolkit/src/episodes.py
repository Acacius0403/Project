import os
from .metadata import load_episodes
from .metadata import load_info
from .loader import get_video_path
def calculate_avg_stats(episodes: list[dict], fps: int) -> dict:
    """
    根据episode列表和FPS，计算平均长度和平均时长
    返回 {"total_episodes": int, "total_frames": int, "avg_length": float, "avg_duration": float}
    """
    total_episodes = len(episodes)
    total_frames = sum(ep["length"] for ep in episodes)
    avg_length = total_frames / total_episodes
    avg_duration = avg_length / fps

    return {"total_episodes": total_episodes, "total_frames": total_frames, "avg_length": avg_length, "avg_duration": avg_duration}

def list_episodes_summary(dataset_path: str) -> list[dict]:
    """
    遍历所有episode，返回每个episode的摘要信息列表
    每个元素包含：pisode_index, task, frames, duration, videos(字符串如"4/4"), statu
    """
    info = load_info(dataset_path)
    episodes = load_episodes(dataset_path)
    cameras = [
        k for k in info["features"].keys()
        if k.startswith("observation.images")
    ]
    fps = info["fps"]

    result = []

    for ep in episodes:
        idx = ep["episode_index"]
        task = ep["tasks"]
        frames = ep["length"]
        duration = frames / fps
        
        video_count = 0

        for cam in cameras:
            vp = get_video_path(dataset_path, idx, cam)
            if os.path.exists(vp):
                video_count += 1

        if video_count < len(cameras):
            status = "Missing videos"
        elif frames < 250:
            status = "Too short"
        elif frames > 350:
            status = "Too long"
        else:
            status = "OK"

        result.append({
            "episode_index": idx,
            "task": task,
            "frames": frames,
            "duration": duration,
            "videos": f"{video_count}/{len(cameras)}",
            "status": status
        })

    return result
    