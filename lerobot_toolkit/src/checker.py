import os
import numpy as np
import cv2
import pandas as pd
from rich.console import Console
from rich.table import Table
from src.metadata import load_info, load_episodes, load_tasks
from src.loader import load_episode_parquet, get_video_path


def run_checks(
    dataset_path: str,
    min_frames: int = 250,
    max_frames: int = 350,
    max_action_jump: float = 0.2,
    max_abs_action: float = 2.0,
    timestamp_tolerance: float = 0.005,
    low_variance_threshold: float = 0.0001,
) -> list[dict]:
    """执行全部 16 项数据质量检查。

    每项检查返回一个 dict: {"check": 检查项名称, "status": "OK"/"WARN"/"ERROR", "detail": 详细描述}

    16 项检查分为三类：
    A类（元信息检查，不依赖 episode 循环）：
      1. meta 目录下 4 个文件是否存在
      2. episode index 是否重复
      3. task 描述是否为空

    B类（逐 episode 检查，需要遍历 30 集）：
      4. parquet 文件是否存在
      5. parquet 行数是否与 episodes.jsonl 一致
      6. 必需字段是否缺失
      7. timestamp 是否单调递增
      8. timestamp 间隔是否接近 1/fps
      9. frame_index 是否从0连续
      10. episode_index 字段值是否正确
      11. state/action 各帧维度是否始终为16
      12. 是否存在 NaN 或 Inf
      13. 是否存在全零 action 帧
      14. 是否存在异常大的 action 跳变
      15. episode 长度是否在正常范围

    C类（视频检查，需要 OpenCV）：
      16. 每路视频是否存在 + 帧数是否匹配
    """
    results = []
    infos = []

    # ========== 1. meta 文件检查 ==========
    for fname in ["info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]:
        fpath = os.path.join(dataset_path, "meta", fname)
        if os.path.exists(fpath):
            results.append({"check": f"Meta: {fname}", "status": "OK", "detail": "存在"})
        else:
            results.append({"check": f"Meta: {fname}", "status": "ERROR", "detail": "缺失"})

    # 加载数据供后续检查
    info = load_info(dataset_path)
    episodes = load_episodes(dataset_path)
    tasks = load_tasks(dataset_path)
    cameras = [k for k in info["features"] if k.startswith("observation.images.")]
    fps = info["fps"]
    expected_interval = 1.0 / fps
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]

    # ========== 2. episode index 重复检查 ==========
    indices = [ep["episode_index"] for ep in episodes]
    if len(indices) == len(set(indices)):  #利用集合自动去重
        results.append({"check": "Episode index uniqueness", "status": "OK", "detail": "无重复"})
    else:
        results.append({"check": "Episode index uniqueness", "status": "ERROR", "detail": "存在重复"})

    # ========== 3. task 描述非空 ==========
    for t in tasks:
        if not t.get("task", "").strip():  #加入,""是为了防止程序崩溃
            results.append({"check": "Task description", "status": "ERROR", "detail": f"task {t.get('task_index')} 描述为空"})
            break
        else:
            results.append({"check": "Task description", "status": "OK", "detail": f"{len(tasks)} 个任务描述完整"})

    # ========== 逐 episode 检查 (第4-16项) ==========
    for ep in episodes:
        idx = ep["episode_index"]
        expected_length = ep["length"]

        # --- 4. parquet 文件存在 ---
        try:
            df = load_episode_parquet(dataset_path, idx)
        except FileNotFoundError:
            results.append({"check": f"Ep {idx} parquet", "status": "ERROR", "detail": "parquet 文件缺失"})
            continue

        # --- 5. 行数与 episodes.jsonl 一致 ---
        actual_length = df.shape[0]  #一行就是一帧
        if actual_length != expected_length:
            results.append({
                "check": f"Ep {idx} row count",
                "status": "WARN",
                "detail": f"parquet {actual_length}行 vs episodes.jsonl {expected_length}行"
            })

        # --- 6. 必需字段 ---
        required_cols = ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            results.append({
                "check": f"Ep {idx} columns",
                "status": "ERROR",
                "detail": f"缺少字段: {missing_cols}"
            })

        # --- 7. timestamp 单调递增 ---
        if not df["timestamp"].is_monotonic_increasing:
            results.append({
                "check": f"Ep {idx} timestamp monotonic",
                "status": "WARN",
                "detail": "timestamp 不单调递增"
            })

        # --- 8. timestamp 间隔检查 ---
        ts_diffs = np.diff(df["timestamp"].to_numpy())
        if len(ts_diffs) > 0:
            max_dev = np.max(np.abs(ts_diffs - expected_interval))
            if max_dev > timestamp_tolerance:
                results.append({
                    "check": f"Ep {idx} timestamp interval",
                    "status": "WARN",
                    "detail": f"间隔偏离最大值 {max_dev:.6f}s (阈值 {timestamp_tolerance})"
                })

        # --- 9. frame_index 连续 ---
        expected_indices = np.arange(actual_length)
        if not np.array_equal(df["frame_index"].to_numpy(), expected_indices):
            results.append({
                "check": f"Ep {idx} frame_index",
                "status": "WARN",
                "detail": "frame_index 不连续或不从0开始"
            })

        # --- 10. episode_index 值正确 ---
        if not (df["episode_index"] == idx).all():
            results.append({
                "check": f"Ep {idx} episode_index value",
                "status": "WARN",
                "detail": "episode_index 字段值不等于当前 episode 编号"
            })

        # --- 11. state/action 维度检查 ---
        state_lens = df["observation.state"].apply(len)  #apply是对每一行做相同的操作，如果len（df[]）算出来的是这里面有几行
        if (state_lens != state_dim).any():   #.any()是对每一行都做判断，如果有一行不等于state_dim，则返回True，否则返回False
            results.append({
                "check": f"Ep {idx} state dim",
                "status": "ERROR",
                "detail": f"存在 state 维度不为 {state_dim}"
            })
        action_lens = df["action"].apply(len)
        if (action_lens != action_dim).any():
            results.append({
                "check": f"Ep {idx} action dim",
                "status": "ERROR",
                "detail": f"存在 action 维度不为 {action_dim}"
            })

        # --- 12. NaN / Inf ---
        state_arr = np.array(df["observation.state"].tolist(), dtype=np.float32)
        action_arr = np.array(df["action"].tolist(), dtype=np.float32)
        nan_state = np.isnan(state_arr).sum()
        inf_state = np.isinf(state_arr).sum()
        nan_action = np.isnan(action_arr).sum()
        inf_action = np.isinf(action_arr).sum()
        nan_total = nan_state + nan_action + inf_state + inf_action
        if nan_total > 0:
            results.append({
                "check": f"Ep {idx} NaN/Inf",
                "status": "ERROR",
                "detail": f"state NaN:{nan_state} Inf:{inf_state}, action NaN:{nan_action} Inf:{inf_action}"
            })

        # --- 13. 全零 action ---
        zero_rows = np.all(np.abs(action_arr) < 1e-10, axis=1)
        if zero_rows.any():
            results.append({
                "check": f"Ep {idx} all-zero action",
                "status": "WARN",
                "detail": f"{zero_rows.sum()} 帧 action 全为零"
            })

        # --- 14. action jump 过大 ---
        if action_arr.shape[0] > 1:  #防止只有一帧时diff报错
            action_diffs = np.abs(np.diff(action_arr, axis=0))
            max_jump_per_frame = np.max(action_diffs, axis=1)  # 每帧取16维中变化最大的
            big_jumps = np.where(max_jump_per_frame > max_action_jump)[0]  #np.where()返回满足条件的下标元组(array([1, 3]),)，故要取出来
            if len(big_jumps) > 0:
                results.append({
                    "check": f"Ep {idx} action jump",
                    "status": "WARN",
                    "detail": f"第 {big_jumps[:5].tolist()} 帧存在较大的 action 跳变"
                })

        # --- 15. episode 长度异常 ---
        if actual_length < min_frames:
            results.append({
                "check": f"Ep {idx} length",
                "status": "WARN",
                "detail": f"过短 {actual_length} 帧 (阈值 {min_frames})"
            })
        elif actual_length > max_frames:
            results.append({
                "check": f"Ep {idx} length",
                "status": "WARN",
                "detail": f"过长 {actual_length} 帧 (阈值 {max_frames})"
            })

        # --- 16. 视频存在 + 帧数匹配 ---
        for cam in cameras:
            vp = get_video_path(dataset_path, idx, cam)
            if not os.path.exists(vp):
                results.append({
                    "check": f"Ep {idx} video: {cam}",
                    "status": "ERROR",
                    "detail": "视频文件缺失"
                })
            else:
                cap = cv2.VideoCapture(vp)
                video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()  #避免占用文件
                if abs(video_frames - actual_length) > 2:  # 允许 ±2 帧误差
                    results.append({
                        "check": f"Ep {idx} video frames: {cam}",
                        "status": "WARN",
                        "detail": f"视频 {video_frames} 帧 vs 数据 {actual_length} 帧"
                    })

    # 汇总
    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    err_count = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\n总计检查 {len(results)} 项: [OK] {ok_count}, [WARN] {warn_count}, [ERROR] {err_count}")

    return results

def write_check_report(results: list[dict], output_path: str) -> None:
    """
    将 run_checks 的结果列表写为 Markdown 格式的检查报告。

    报告包含：状态表格 + 汇总统计（OK/WARN/ERROR 数量）。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 数据质量检查报告\n\n")
        f.write(f"检查时间: {pd.Timestamp.now()}\n\n")

        f.write("| Status | Check | Detail |\n")
        f.write("|--------|-------|--------|\n")
        for r in results:
            emoji = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}.get(r["status"], "")
            f.write(f"| {emoji} {r['status']} | {r['check']} | {r['detail']} |\n")

        f.write("\n## 汇总\n\n")
        ok = sum(1 for r in results if r["status"] == "OK")
        warn = sum(1 for r in results if r["status"] == "WARN")
        err = sum(1 for r in results if r["status"] == "ERROR")
        f.write(f"- ✅ OK: {ok} 项\n")
        f.write(f"- ⚠️ WARN: {warn} 项\n")
        f.write(f"- ❌ ERROR: {err} 项\n")

    print(f"检查报告已保存到 {output_path}")

def print_rich_summary(results: list[dict]):
    """
    使用 Rich 在终端打印美观的检查结果表格
    """
    # 创建总表
    table = Table(title="[red bold]Data Quality Check Summary[/red bold]", show_header=True, header_style="bold magenta")
    table.add_column("State", justify="center", style="bold", width=10)
    table.add_column("Item", style="dim", width=50)
    table.add_column("Detail", overflow="fold")

    # 颜色映射
    status_colors = {
        "OK": "green",
        "WARN": "yellow",
        "ERROR": "red"
    }
    icons = {
        "OK": "✅",
        "WARN": "⚠️",
        "ERROR": "❌"
    }

    # 逐行添加
    for res in results:
        status = res["status"]
        color = status_colors[status]
        table.add_row(
            f"[{color}]{icons[status]} {status}[/{color}]",
            res["check"],
            res["detail"]
        )

    console = Console()
    # 打印表格
    console.print("\n")
    console.print(table)
    console.print("\n")

    # 打印汇总统计
    ok = sum(1 for r in results if r["status"] == "OK")
    warn = sum(1 for r in results if r["status"] == "WARN")
    err = sum(1 for r in results if r["status"] == "ERROR")
    
    console.print(f"✅ [green]OK: {ok} items[/green]")
    console.print(f"⚠️ [yellow]Warn: {warn} items[/yellow]")
    console.print(f"❌ [red]Error: {err} items[/red]")
    console.print(f"📌 Sum: {len(results)} items\n", style="bold blue")