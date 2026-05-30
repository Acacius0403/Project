"""Gradio web UI — multi-camera sync, state curves, quality checks, reports."""

import os
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import gradio as gr
from src.metadata import load_info, load_episodes, load_tasks
from src.loader import load_episode_parquet, get_video_path
from src.episodes import calculate_avg_stats
from src.checker import run_checks, write_check_report
from src.report import generate_report
from src.statistics import compute_stats, save_stats_json, generate_figures


# ============================================================
# 全局状态变量
# ============================================================
_DATASET_PATH = None
_INFO = None
_EPISODES = None
_CAMERAS = None
_STATE_NAMES = None
_CURRENT_EPISODE = None
_CURRENT_STATES = None
_CURRENT_TIMESTAMPS = None


def reset_episode_state():
    global _CURRENT_EPISODE, _CURRENT_STATES, _CURRENT_TIMESTAMPS
    _CURRENT_EPISODE = None
    _CURRENT_STATES = None
    _CURRENT_TIMESTAMPS = None


# ============================================================
# 数据集加载
# ============================================================
def do_load_dataset(dataset_path):
    global _DATASET_PATH, _INFO, _EPISODES, _CAMERAS, _STATE_NAMES
    reset_episode_state()

    if not dataset_path or not os.path.isdir(dataset_path):
        return "❌ Path not found.", gr.update(choices=[]), gr.update(minimum=0, maximum=1, value=0), None, None, None, None, None
    try:
        _DATASET_PATH = dataset_path
        _INFO = load_info(dataset_path)
        _EPISODES = load_episodes(dataset_path)
        _CAMERAS = [k for k in _INFO["features"] if k.startswith("observation.images.")]
        _STATE_NAMES = _INFO["features"]["observation.state"]["names"]
        tasks = load_tasks(dataset_path)
        summary = calculate_avg_stats(_EPISODES, _INFO["fps"])

        status = (
            f"✅ {_INFO['robot_type']} | {_INFO['codebase_version']} | "
            f"{summary['total_episodes']} eps | {summary['total_frames']} frames | "
            f"{_INFO['fps']} fps | {len(_CAMERAS)} cameras\n"
            f"Task: {tasks[0]['task']}"
        )

        choices = [f"EP{ep['episode_index']}" for ep in _EPISODES]
        return status, gr.update(choices=choices), gr.update(minimum=0, maximum=1, value=0), None, None, None, None, None
    except Exception as e:
        traceback.print_exc()
        return f"❌ {e}", gr.update(choices=[]), gr.update(minimum=0, maximum=1, value=0), None, None, None, None, None


# ============================================================
# 加载 Episode
# ============================================================
def load_episode_and_update(ep_str):
    global _CURRENT_EPISODE, _CURRENT_STATES, _CURRENT_TIMESTAMPS
    reset_episode_state()

    if not _DATASET_PATH or not ep_str:
        return gr.update(minimum=0, maximum=1, value=0), None, None, None, None, None

    try:
        # 修复：兼容列表/字符串两种情况
        if isinstance(ep_str, list):
            ep_str = ep_str[0] if len(ep_str) > 0 else ""
            
        ep_idx = int(ep_str.replace("EP", ""))
        df = load_episode_parquet(_DATASET_PATH, ep_idx)
        _CURRENT_EPISODE = ep_idx
        _CURRENT_STATES = np.array(df["observation.state"].tolist(), dtype=np.float32)
        _CURRENT_TIMESTAMPS = df["timestamp"].to_numpy(dtype=np.float32)
        total = max(1, len(_CURRENT_STATES) - 1)

        imgs, fig = render_frame(0)
        return gr.update(minimum=0, maximum=total, value=0), *imgs, fig
    except Exception as e:
        traceback.print_exc()
        return gr.update(minimum=0, maximum=1, value=0), None, None, None, None, None


def render_frame(frame_idx):
    if _CURRENT_STATES is None:
        blank = np.full((240, 320, 3), 40, dtype=np.uint8)
        return (blank, blank, blank, blank), None

    total = len(_CURRENT_STATES) - 1
    frame_idx = max(0, min(total, frame_idx))
    ts = _CURRENT_TIMESTAMPS[frame_idx]

    camera_imgs = []
    for cam in _CAMERAS:
        vp = get_video_path(_DATASET_PATH, _CURRENT_EPISODE, cam)
        if not os.path.exists(vp):
            camera_imgs.append(np.full((240, 320, 3), 50, dtype=np.uint8))
            continue
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        cap.release()
        if ret:
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.full((240, 320, 3), 30, dtype=np.uint8)
        camera_imgs.append(img)

    x = np.arange(len(_CURRENT_STATES))
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for j in range(8):
        axes[0].plot(x, _CURRENT_STATES[:, j], color=colors[j], lw=0.6, label=_STATE_NAMES[j].replace(".pos", ""))
    axes[0].axvline(frame_idx, color="red", lw=2, linestyle="--")
    axes[0].set_ylabel("Left Arm")
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for j in range(8, 16):
        axes[1].plot(x, _CURRENT_STATES[:, j], color=colors[j-8], lw=0.6, label=_STATE_NAMES[j].replace(".pos", ""))
    axes[1].axvline(frame_idx, color="red", lw=2, linestyle="--")
    axes[1].set_ylabel("Right Arm")
    axes[1].set_xlabel("Frame")
    axes[1].legend(fontsize=6, ncol=2, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.close(fig)
    return tuple(camera_imgs), fig


def on_slide_frame(frame_idx):
    try:
        imgs, fig = render_frame(int(frame_idx))
        return (*imgs, fig)
    except:
        traceback.print_exc()
        return None, None, None, None, None


# ============================================================
# Checks & Report
# ============================================================
def do_checks():
    if not _DATASET_PATH:
        return "❌ Load dataset first"
    try:
        res = run_checks(_DATASET_PATH)
        md = "# Quality Check\n| Status | Check | Detail |\n|---|---|---|\n"
        for r in res:
            e = {"OK":"✅","WARN":"⚠️","ERROR":"❌"}.get(r["status"],"")
            md += f"|{e}|{r['check']}|{r['detail']}|\n"
        return md
    except:
        return "❌ Error"

def do_report():
    if not _DATASET_PATH:
        return None
    try:
        tmp = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp, exist_ok=True)
        generate_report(_DATASET_PATH, os.path.join(tmp, "report.md"))
        return os.path.join(tmp, "report.md")
    except:
        return None


# ============================================================
# UI
# ============================================================
def create_ui():
    with gr.Blocks(title="LeRobot Toolkit") as demo:
        gr.Markdown("# 🤖 LeRobot Dataset Toolkit")

        with gr.Row():
            ds_input = gr.Textbox(label="Dataset Path", value="../dataset_0423_v2.1")
            load_btn = gr.Button("🔍 Load Dataset", variant="primary")
        status_box = gr.Textbox(label="Status", lines=2)

        with gr.Row():
            # 修复：强制使用下拉菜单样式，不展开
            ep_dd = gr.Dropdown(
                label="Episode",
                choices=[],
                allow_custom_value=True,
                container=True
            )
            load_ep_btn = gr.Button("📥 Load Episode")
        frame_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label="Frame")

        gr.Markdown("### Cameras")
        with gr.Row():
            c1 = gr.Image(width=320, height=240)
            c2 = gr.Image(width=320, height=240)
        with gr.Row():
            c3 = gr.Image(width=320, height=240)
            c4 = gr.Image(width=320, height=240)

        plot = gr.Plot(label="State Curves")

        with gr.Row():
            chk = gr.Button("🔎 Check Quality")
            rep = gr.Button("📄 Report")
        check_out = gr.Markdown()
        rep_out = gr.File()

        # Events
        load_btn.click(
            do_load_dataset, ds_input,
            [status_box, ep_dd, frame_slider, c1, c2, c3, c4, plot]
        )
        load_ep_btn.click(
            load_episode_and_update, ep_dd,
            [frame_slider, c1, c2, c3, c4, plot]
        )
        frame_slider.change(on_slide_frame, frame_slider, [c1, c2, c3, c4, plot])
        chk.click(do_checks, outputs=check_out)
        rep.click(do_report, outputs=rep_out)

    return demo


def launch_web(dataset_path="../dataset_0423_v2.1", share=False):
    demo = create_ui()
    demo.launch(share=share, inbrowser=True)