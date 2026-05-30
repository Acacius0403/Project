import numpy as np
import json
import os
import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from .loader import load_episode_parquet
from .metadata import load_episodes, load_info

def compute_stats(dataset_path: str, low_variance_threshold: float = 0.0001) -> dict:
    """
    遍历所有episode，拼接state和action矩阵，计算各维度统计值
    返回一个大字典，包含：
    - state: 每维度的 min/max/mean/std + NaN计数 + 低方差维度列表
    - action: 同上 + 全零帧数 + 相邻帧变化量分布
    - episode_stats: episode 长度列表、总数、总帧数
    """
    episodes = load_episodes(dataset_path)
    info = load_info(dataset_path)
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]
    state_names = info["features"]["observation.state"]["names"]

    all_states = []
    all_actions = []

    for ep in episodes:
        idx = ep["episode_index"]
        df = load_episode_parquet(dataset_path, idx)

        """
        df 的形状是：
        0    [1, 2]
        1    [3, 4]
        2    [5, 6]

        经过下面的tolist变为：
        [
            [1, 2],   # 第 0 帧
            [3, 4],   # 第 1 帧
            [5, 6]    # 第 2 帧
        ]

        再经过np.array变为：
        array([
            [1., 2.],
            [3., 4.],
            [5., 6.]
        ])
        """

        states = np.array(df["observation.state"].tolist())
        actions = np.array(df["action"].tolist())

        all_states.append(states)
        all_actions.append(actions)
    """
    A = [[1,2],
        [3,4]]  (2行2列)

    B = [[5,6],
        [7,8]]  (2行2列)

    拼接后(axis=0)
    [[1,2],
    [3,4],
    [5,6],
    [7,8]]

    每一列就是一个特征，也就是state_names里面的16个维度

    axis=1
    [[1,2, 5,6],
    [3,4, 7,8]]
    """

    state_matrix = np.concatenate(all_states, axis=0)  # (8369, 16)
    action_matrix = np.concatenate(all_actions, axis=0)  # (8369, 16)

    stats = {
        "state": {},
        "action": {},
        "episode_stats": {
            "count": len(episodes),
            "total_frames": int(state_matrix.shape[0]),
            "lengths": [ep["length"] for ep in episodes],
        },
    }

    for label, matrix, dim_names in [
        ("state", state_matrix, state_names),
        ("action", action_matrix, state_names), 
    ]:
        for i, name in enumerate(dim_names):
            col = matrix[:, i] 
            stats[label][name] = {
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }

        zero_rows = np.all(np.abs(matrix) < 1e-10, axis=1)
        stats[label]["_all_zero_rows"] = int(np.sum(zero_rows))

        stats[label]["_nan_count"] = int(np.sum(np.isnan(matrix)))   #np.isnan 返回一个布尔矩阵，是NaN变成True，True加起来就是NaN的个数
        stats[label]["_inf_count"] = int(np.sum(np.isinf(matrix)))

    action_diff = np.diff(action_matrix, axis=0)  # (8368, 16)
    stats["action"]["_diff_mean"] = float(np.mean(np.abs(action_diff)))
    stats["action"]["_diff_max"] = float(np.max(np.abs(action_diff)))
    stats["action"]["_diff_per_dim"] = {
        state_names[i]: {
            "mean_abs_diff": float(np.mean(np.abs(action_diff[:, i]))),
            "max_abs_diff": float(np.max(np.abs(action_diff[:, i]))),
        }
        for i in range(action_dim)
    }

    for label, matrix in [("state", state_matrix), ("action", action_matrix)]:
        variances = np.var(matrix, axis=0)  # (16,)
        low_var_mask = variances < low_variance_threshold  # 正确：得到布尔数组
        stats[label]["_low_variance_dims"] = [
            state_names[i] for i, is_low in enumerate(low_var_mask) if is_low
    ]

    return stats, action_matrix
        
def generate_figures(stats: dict, action_matrix: np.ndarray, output_dir: str) -> list[str]: 
    """
    根据 compute_stats 返回的统计字典，生成 3 张 PNG 图表。

    图1: episode 长度分布直方图
    图2: action 各维度 min-max 范围图（水平条形图）
    图3: action 各维度平均变化量图（柱状图）

    返回保存的图片路径列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    state_names = list(stats["state"].keys())
    # 去掉以下划线开头的统计键, which are 辅助统计字段，不是最终要呈现的
    dim_names = [n for n in state_names if not n.startswith("_")]
    n_dims = len(dim_names)

    # --- 图 1: Episode 长度 水平条形图 ---
    fig, ax = plt.subplots(figsize=(11, 8))
    lengths = stats["episode_stats"]["lengths"]
    episode_indices = np.arange(len(lengths))

    # 按阈值区分柱子颜色
    bar_colors = []
    for val in lengths:
        if val < 250:
            bar_colors.append("#e74c3c")    # 太短红色
        elif val > 350:
            bar_colors.append("#f39c12")    # 太长黄色
        else:
            bar_colors.append("#4C72B0")    # 正常蓝色

    bars = ax.barh(episode_indices, lengths, height=0.7, color=bar_colors, alpha=0.85)

    # 阈值竖线
    ax.axvline(250, color="darkred", linestyle="--", linewidth=1.5, label="Min threshold (250)")
    ax.axvline(350, color="darkorange", linestyle="--", linewidth=1.5, label="Max threshold (350)")

    # 柱子末端标注帧数数字
    for idx, val in enumerate(lengths):
        ax.text(val + 2, idx, str(val), va="center", fontsize=8)

    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Episode Index", fontsize=12)
    ax.set_title("Episode Length per Episode", fontsize=14, pad=15)
    ax.legend(fontsize=10)
    ax.set_yticks(episode_indices)
    ax.set_xlim(0, max(lengths) * 1.1)  # 留出文字空间

    plt.tight_layout()
    path1 = os.path.join(output_dir, "episode_length_barh.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path1)

    # --- 图 2: Action 维度分布山脊图（带名称下方标注 + 范围修正） ---
    fig, axes = plt.subplots(nrows=n_dims, figsize=(10, 11), sharex=True)
    fig.subplots_adjust(hspace=-0.4)

    # 先拿到所有维度的全局 min/max，用于统一 x 轴范围，避免分布跑出边界
    all_mins = [stats["action"][n]["min"] for n in dim_names]
    all_maxs = [stats["action"][n]["max"] for n in dim_names]
    x_min = min(all_mins) - 0.1  # 留一点边距
    x_max = max(all_maxs) + 0.1

    for i, name in enumerate(dim_names):
        data = action_matrix[:, i]
        min_val = stats["action"][name]["min"]
        max_val = stats["action"][name]["max"]
        mean_val = stats["action"][name]["mean"]

        # 1. 画密度曲线
        sns.kdeplot(
            data,
            ax=axes[i],
            fill=True,
            alpha=0.85,
            color="#5faef8",
            linewidth=1,
            clip=(min_val, max_val)  # 关键：强制 KDE 只在 [min, max] 范围内绘制
        )

        # 2. 设置该子图的 x 轴范围，避免分布跑出边界
        axes[i].set_xlim(x_min, x_max)

        # 3. 隐藏多余坐标轴
        axes[i].set_yticks([])
        axes[i].set_ylabel("")
        axes[i].spines[["left", "right", "top"]].set_visible(False)

        # 4. 在维度名称下方标注 [min:mean:max]
        # 先画维度名称
        axes[i].text(
            -0.02, 0.6, name,
            transform=axes[i].transAxes,
            ha="right", va="center", fontsize=9
        )
        # 再在名称下方画数值
        axes[i].text(
            -0.02, 0.2,
            f"[{min_val:.2f} : {mean_val:.2f} : {max_val:.2f}]",
            transform=axes[i].transAxes,
            ha="right", va="center", fontsize=8, color="#888888"
        )

    axes[-1].set_xlabel("Action Value", fontsize=12)
    fig.suptitle("Action Distribution per Dimension (Ridgeline)", fontsize=14, y=0.98)
    plt.tight_layout()
    path2 = os.path.join(output_dir, "action_ridgeline.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path2)

    # --- 图 3: Mean Absolute Action Change (点线图) ---
    if "_diff_per_dim" in stats["action"]:
        diff_data = stats["action"]["_diff_per_dim"]
        fig, ax = plt.subplots(figsize=(12, 5))
        mean_diffs = [diff_data[n]["mean_abs_diff"] for n in dim_names]
        x_pos = range(n_dims)
        
        # 画点线图
        ax.plot(x_pos, mean_diffs, marker='o', linestyle='-', color="#4C72B0", linewidth=1.5, markersize=5)
        
        # 标注每个点的具体数值（保留4位小数）
        for x, y in zip(x_pos, mean_diffs):
            ax.text(x, y, f"{y:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(dim_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Absolute Action Change")
        ax.set_title("Mean Action Change per Dimension")
        fig.tight_layout()
        path3 = os.path.join(output_dir, "action_change_line.png")
        fig.savefig(path3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path3)

    return saved

def save_stats_json(stats: dict, output_path: str) -> None:
    """把统计结果存为 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 把 _diff_per_dim 这种深层嵌套简化一下（可选）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


