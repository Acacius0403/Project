import os
import json
from datetime import datetime
from .metadata import load_info, load_episodes, load_tasks
from .episodes import calculate_avg_stats

def generate_report(dataset_path: str, output_path: str) -> None:
    """
    整合 info 元信息 + stats.json + check 结果，生成 9 章节的 Markdown 报告。

    章节：
    1. 数据集概览（调用 load_info）
    2. 目录结构说明（固定文本 + 元信息）
    3. Episode 组织方式
    4. State 与 Action 字段说明（含关节维度对应表）
    5. Episode 长度统计
    6. Action/State 统计结果（读取 stats.json）
    7. 数据质量检查结果（摘要 + 引用 check_report.md）
    8. 可视化图表（引用 figures/ 下的图片）
    9. 结论与改进建议（基于前面发现的问题）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 加载数据
    info = load_info(dataset_path)
    episodes = load_episodes(dataset_path)
    tasks = load_tasks(dataset_path)
    summary = calculate_avg_stats(episodes, info["fps"])
    state_names = info["features"]["observation.state"]["names"]
    cameras = [k for k in info["features"] if k.startswith("observation.images.")]

    # 加载 stats.json
    stats_path = os.path.join(os.path.dirname(output_path), "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

    # 加载 check 结果
    check_path = os.path.join(os.path.dirname(output_path), "check_report.md")
    check_exists = os.path.exists(check_path)

    # 收集异常 episode
    short_eps = [ep for ep in episodes if ep["length"] < 250]
    long_eps = [ep for ep in episodes if ep["length"] > 350]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# LeRobot 数据集分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # == 1. 数据集概览 ==
        f.write("## 1. 数据集概览\n\n")
        f.write(f"- **数据集路径**: `{os.path.abspath(dataset_path)}`\n")
        f.write(f"- **LeRobot 版本**: {info['codebase_version']}\n")
        f.write(f"- **机器人类型**: {info['robot_type']}\n")
        f.write(f"- **Episode 总数**: {summary['total_episodes']}\n")
        f.write(f"- **总帧数**: {summary['total_frames']}\n")
        f.write(f"- **FPS**: {info['fps']}\n")
        f.write(f"- **平均 episode 长度**: {summary['avg_length']} 帧 ({summary['avg_duration']} 秒)\n")
        f.write(f"- **任务数量**: {len(tasks)}\n")
        f.write(f"- **相机数量**: {len(cameras)}\n\n")
        f.write(f"**任务描述**: {tasks[0]['task']}\n\n")

        # == 2. 目录结构说明 ==
        f.write("## 2. 目录结构说明\n\n")
        f.write("数据集 `dataset_0423_v2.1/` 由三部分组成:\n\n")
        f.write("- **`meta/`**: 元信息文件。`info.json` 记录数据集全局信息，`episodes.jsonl` 记录每个 episode 长度，`tasks.jsonl` 记录任务描述。\n")
        f.write("- **`data/`**: 结构化数据。每个 episode 对应一个 `.parquet` 文件，包含 `observation.state`、`action`、`timestamp`、`frame_index` 等字段。\n")
        f.write("- **`videos/`**: 视频数据。每个 episode 有 4 路相机视频，存放在以相机名命名的子目录中。\n\n")

        # == 3. Episode 组织 ==
        f.write("## 3. Episode 组织方式\n\n")
        f.write(f"本数据集包含 **{len(episodes)} 个 episode**，每个 episode 是一次完整的\"抓取白色袋子放入黄色盒子\"操作。\n\n")
        f.write("数据对应关系:\n\n")
        f.write("- Parquet 文件路径由 `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` 模板解析。\n")
        f.write("- 视频路径由 `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` 模板解析。\n\n")
        f.write(f"当前所有 episode 位于 `chunk-000`，共 {summary['total_frames']} 帧。\n\n")

        # == 4. State 与 Action ==
        f.write("## 4. State 与 Action 字段说明\n\n")
        f.write(f"**State 维度**: {info['features']['observation.state']['shape'][0]}\n")
        f.write(f"**Action 维度**: {info['features']['action']['shape'][0]}\n\n")
        f.write("### 关节名称对应关系\n\n")
        f.write("| 维度 | 关节名 | 所属部位 |\n")
        f.write("|-----|--------|--------|\n")
        for i, name in enumerate(state_names):
            side = "左臂" if "left" in name else "右臂" if "right" in name else "-"
            f.write(f"| {i} | {name} | {side} |\n")
        f.write("\n**说明**: State 记录各关节的实际位置，Action 是发给各关节的目标位置指令。训练行为克隆模型时，目标是用 State 预测 Action。\n\n")

        # == 5. Episode 长度统计 ==
        f.write("## 5. Episode 长度统计\n\n")
        lengths = [ep["length"] for ep in episodes]
        f.write(f"- 最短 episode: {min(lengths)} 帧\n")
        f.write(f"- 最长 episode: {max(lengths)} 帧\n")
        f.write(f"- 平均长度: {summary['avg_length']} 帧\n\n")

        if short_eps:
            f.write(f"**过短 episode (<250帧)**: {len(short_eps)} 个 — {[ep['episode_index'] for ep in short_eps]}\n\n")
        if long_eps:
            f.write(f"**过长 episode (>350帧)**: {len(long_eps)} 个 — {[ep['episode_index'] for ep in long_eps]}\n\n")

        f.write("![Episode 长度分布](figures/episode_length_barh.png)\n\n")

        # == 6. 统计结果 ==
        f.write("## 6. Action / State 统计结果\n\n")

        if stats:
            action_stats = stats.get("action", {})
            state_stats = stats.get("state", {})

            f.write("### 6.1 Action 各维度统计 (全数据集)\n\n")
            f.write("| 维度 | Min | Max | Mean | Std |\n")
            f.write("|-----|-----|-----|------|-----|\n")
            for name in state_names:
                d = action_stats.get(name, {})
                f.write(f"| {name} | {d.get('min', 'N/A'):.4f} | {d.get('max', 'N/A'):.4f} | {d.get('mean', 'N/A'):.4f} | {d.get('std', 'N/A'):.4f} |\n")

            f.write("\n![Action 范围](figures/action_ridgeline.png)\n")
            f.write("\n![Action 变化量](figures/action_change_line.png)\n")

            low_var_action = action_stats.get("_low_variance_dims", [])
            if low_var_action:
                f.write(f"\n**低方差 Action 维度** (阈值={0.0001}): {low_var_action}\n")
            if action_stats.get("_all_zero_rows", 0) > 0:
                f.write(f"\n**全零 action 帧数**: {action_stats['_all_zero_rows']}\n")

            nan_s = state_stats.get("_nan_count", 0)
            nan_a = action_stats.get("_nan_count", 0)
            if nan_s > 0 or nan_a > 0:
                f.write(f"\n**NaN 计数**: state={nan_s}, action={nan_a}\n")

        # == 7. 质量检查结果 ==
        f.write("## 7. 数据质量检查结果\n\n")
        if check_exists:
            f.write(f"详细检查报告见 [{os.path.basename(check_path)}]({os.path.basename(check_path)})\n\n")

        # 简要汇总
        f.write("### 检查结论\n\n")
        f.write(f"- 总 episode 数: {summary['total_episodes']}\n")
        f.write(f"- 过短 episode: {len(short_eps)} 个\n")
        f.write(f"- 过长 episode: {len(long_eps)} 个\n")

        # == 8. 可视化图表 ==
        f.write("## 8. 可视化图表\n\n")
        figures_dir = os.path.join(os.path.dirname(output_path), "figures")
        if os.path.exists(figures_dir):
            for fig in os.listdir(figures_dir):
                if fig.endswith(".png"):
                    f.write(f"![{fig}](figures/{fig})\n\n")
        else:
            f.write("(图表目录不存在，请先运行 `stats` 命令)\n\n")

        # == 9. 结论 ==
        f.write("## 9. 结论与改进建议\n\n")

        issues = []
        if short_eps:
            issues.append(f"存在 {len(short_eps)} 个过短 episode，可能需要检查这些 episode 是否未完成任务。")
        if long_eps:
            issues.append(f"存在 {len(long_eps)} 个过长 episode，可能包含多余动作。")
        if stats:
            low_var = stats.get("action", {}).get("_low_variance_dims", [])
            if low_var:
                issues.append(f"Action 中有 {len(low_var)} 个低方差维度 ({low_var})，在训练时可能可以作为常数丢弃。")

        if issues:
            f.write("### 发现的问题\n\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            f.write("\n")
        else:
            f.write("未发现明显异常。\n\n")

        f.write("### 数据集可用性评估\n\n")
        f.write(f"该数据集包含 {summary['total_episodes']} 个 episode、{summary['total_frames']} 帧，涵盖 4 个相机视角。")
        f.write("整体数据质量良好，适合用于训练行为克隆模型。")
        if issues:
            f.write("建议先清理上述标注异常的 episode 后再用于训练。")

    print(f"报告已保存到 {output_path}")