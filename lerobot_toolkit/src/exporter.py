import os
import json
import shutil
from .metadata import load_info, load_episodes, load_tasks


def export_episodes(dataset_path: str, episode_indices: list[int], output_path: str) -> None:
    """
    把用户指定的几个 episode 导出为独立的子集目录

    输入：数据集路径 + 要导出的 episode 编号列表 + 输出目录
    输出：一个新的 LeRobot 格式子目录
    """
    info = load_info(dataset_path)
    episodes = load_episodes(dataset_path)
    cameras = [k for k in info["features"] if k.startswith("observation.images.")]

    chunk_idx = 0  # subset always goes into chunk-000

    # 1. Create output directory structure
    os.makedirs(os.path.join(output_path, "meta"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "data", f"chunk-{chunk_idx:03d}"), exist_ok=True)
    for cam in cameras:
        os.makedirs(os.path.join(output_path, "videos", f"chunk-{chunk_idx:03d}", cam), exist_ok=True)

    # 2. Copy selected parquet files
    total_frames = 0
    for ep in episodes:
        idx = ep["episode_index"]
        if idx not in episode_indices:
            continue

        # Copy parquet
        src_parquet = os.path.join(
            dataset_path, "data",
            f"chunk-{idx // info['chunks_size']:03d}",
            f"episode_{idx:06d}.parquet"
        )
        dst_parquet = os.path.join(
            output_path, "data", f"chunk-{chunk_idx:03d}",
            f"episode_{idx:06d}.parquet"
        )
        shutil.copy2(src_parquet, dst_parquet)
        total_frames += ep["length"]

        # Copy 4 camera videos
        for cam in cameras:
            src_video = os.path.join(
                dataset_path, "videos",
                f"chunk-{idx // info['chunks_size']:03d}",
                cam,
                f"episode_{idx:06d}.mp4"
            )
            dst_video = os.path.join(
                output_path, "videos", f"chunk-{chunk_idx:03d}",
                cam,
                f"episode_{idx:06d}.mp4"
            )
            if os.path.exists(src_video):
                shutil.copy2(src_video, dst_video)
            else:
                print(f"  [WARN] 缺失视频: {src_video}")

    # 3. Generate new info.json
    new_info = {
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "total_episodes": len(episode_indices),
        "total_frames": total_frames,
        "total_tasks": info["total_tasks"],
        "chunks_size": info["chunks_size"],
        "fps": info["fps"],
        "splits": {"train": f"0:{len(episode_indices)}"},
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "features": info["features"],
        "total_chunks": 1,
        "total_videos": len(episode_indices) * len(cameras),
    }
    with open(os.path.join(output_path, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(new_info, f, indent=2, ensure_ascii=False)

    # 4. Write filtered episodes.jsonl
    with open(os.path.join(output_path, "meta", "episodes.jsonl"), "w", encoding="utf-8") as f:
        for ep in episodes:
            if ep["episode_index"] in episode_indices:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # 5. Copy tasks.jsonl as-is
    shutil.copy2(
        os.path.join(dataset_path, "meta", "tasks.jsonl"),
        os.path.join(output_path, "meta", "tasks.jsonl")
    )

    print(f"导出完成: {len(episode_indices)} episodes, {total_frames} frames → {output_path}")
