import cv2
import numpy as np
from .loader import load_episode_parquet, get_video_path
from .metadata import load_info


def replay_episode(dataset_path: str, episode_idx: int) -> None:
    """
    使用 OpenCV 视频窗口回放一段数据，并叠加显示状态 / 动作信息。

    键盘控制：
    空格        — 播放 / 暂停自动播放
    右箭头 / D  — 下一帧
    左箭头 / A  — 上一帧
    1-4         — 切换摄像头视角（1=顶部01, 2=顶部02, 3=左手腕, 4=右手腕）
    Q / ESC     — 退出
    J           — 跳转到指定帧（在终端输入帧序号）
    """
    df = load_episode_parquet(dataset_path, episode_idx)
    info = load_info(dataset_path)
    state_names = info["features"]["observation.state"]["names"]
    total_frames = len(df)
    fps = info["fps"]

    cameras = [
        ("observation.images.top01",       "Top 01"),
        ("observation.images.top02",       "Top 02"),
        ("observation.images.left_wrist",  "Left Wrist"),
        ("observation.images.right_wrist", "Right Wrist"),
    ]

    states = np.array(df["observation.state"].tolist(), dtype=np.float32)
    actions = np.array(df["action"].tolist(), dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float32)

    # Open all 4 video captures
    caps = []
    for cam_name, _ in cameras:
        path = get_video_path(dataset_path, episode_idx, cam_name)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {path}")
        caps.append(cap)

    frame_idx = 0
    playing = True
    window_name = f"Episode {episode_idx} — 4-Camera Sync  |  {total_frames} frames  |  ~{total_frames/fps:.1f}s"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 780)

    CELL_W, CELL_H = 480, 360  # each cell
    GRID_W, GRID_H = CELL_W * 2, CELL_H * 2
    INFO_H = 60  # bottom info bar

    def read_frame(cap, idx):
        """Read frame at given index from a VideoCapture."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (CELL_W, CELL_H))
        return frame

    def build_grid(i):
        """Assemble 2×2 camera grid + bottom info bar."""
        frames = [read_frame(c, i) for c in caps]

        # 2×2 grid
        row1 = np.hstack([frames[0], frames[1]])
        row2 = np.hstack([frames[2], frames[3]])
        grid = np.vstack([row1, row2])  # (720, 960, 3)

        # Camera labels on each cell
        for j, (_, label) in enumerate(cameras):
            col, row = j % 2, j // 2
            x, y = col * CELL_W + 5, row * CELL_H + 25
            cv2.putText(grid, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

        # Frame counter on top-left
        ts = timestamps[i]
        txt = f"Frame {i:04d}/{total_frames-1}  |  t = {ts:.3f}s  |  {'PLAYING' if playing else 'PAUSED'}"
        cv2.putText(grid, txt, (10, CELL_H * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)

        # Bottom info panel
        info_panel = np.zeros((INFO_H, GRID_W, 3), dtype=np.uint8)
        state = states[i]
        action = actions[i]
        delta = np.abs(action - state)

        # Show top 4 joints with largest delta
        top4 = np.argsort(delta)[::-1][:4]
        parts = []
        for j in top4:
            parts.append(f"{state_names[j].split('.')[0]}: s={state[j]:+.4f} a={action[j]:+.4f}")
        cv2.putText(info_panel, " | ".join(parts), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Help bar
        help_text = "SPACE:Play/Pause | <- ->:Prev/Next | J:Jump | Q:Quit"
        cv2.putText(info_panel, help_text, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return np.vstack([grid, info_panel])

    print(f"\nEpisode {episode_idx}  |  {total_frames} frames  |  ~{total_frames/fps:.1f}s")
    print("4 cameras displayed in 2×2 layout.")
    print("Controls: SPACE=play/pause | <- -> =prev/next | J=jump | Q=quit\n")

    while True:
        canvas = build_grid(frame_idx)
        cv2.imshow(window_name, canvas)

        delay = int(1000 / fps) if playing else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            playing = not playing
        elif key == 81 or key == ord("a"):   # Left / A
            playing = False
            frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == ord("d"):   # Right / D
            playing = False
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord("j"):
            playing = False
            target = input(f"Jump to frame (0-{total_frames-1}): ").strip()
            if target.isdigit():
                frame_idx = max(0, min(total_frames - 1, int(target)))
        elif playing:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = 0  # loop

    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    print("Replay finished.")