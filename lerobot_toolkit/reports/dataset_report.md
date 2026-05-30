# LeRobot 数据集分析报告

**生成时间**: 2026-05-24 21:32:30

---

## 1. 数据集概览

- **数据集路径**: `C:\Users\Acacius\Desktop\embodied\dataset_0423_v2.1`
- **LeRobot 版本**: v2.1
- **机器人类型**: oli
- **Episode 总数**: 30
- **总帧数**: 8369
- **FPS**: 30
- **平均 episode 长度**: 278.96666666666664 帧 (9.298888888888888 秒)
- **任务数量**: 1
- **相机数量**: 4

**任务描述**: Pick up the white bag in front of you and put it in the yellow box in front of you.

## 2. 目录结构说明

数据集 `dataset_0423_v2.1/` 由三部分组成:

- **`meta/`**: 元信息文件。`info.json` 记录数据集全局信息，`episodes.jsonl` 记录每个 episode 长度，`tasks.jsonl` 记录任务描述。
- **`data/`**: 结构化数据。每个 episode 对应一个 `.parquet` 文件，包含 `observation.state`、`action`、`timestamp`、`frame_index` 等字段。
- **`videos/`**: 视频数据。每个 episode 有 4 路相机视频，存放在以相机名命名的子目录中。

## 3. Episode 组织方式

本数据集包含 **30 个 episode**，每个 episode 是一次完整的"抓取白色袋子放入黄色盒子"操作。

数据对应关系:

- Parquet 文件路径由 `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet` 模板解析。
- 视频路径由 `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` 模板解析。

当前所有 episode 位于 `chunk-000`，共 8369 帧。

## 4. State 与 Action 字段说明

**State 维度**: 16
**Action 维度**: 16

### 关节名称对应关系

| 维度 | 关节名 | 所属部位 |
|-----|--------|--------|
| 0 | left_shoulder_pitch_joint.pos | 左臂 |
| 1 | left_shoulder_roll_joint.pos | 左臂 |
| 2 | left_shoulder_yaw_joint.pos | 左臂 |
| 3 | left_elbow_joint.pos | 左臂 |
| 4 | left_wrist_yaw_joint.pos | 左臂 |
| 5 | left_wrist_pitch_joint.pos | 左臂 |
| 6 | left_wrist_roll_joint.pos | 左臂 |
| 7 | left_claw_joint.pos | 左臂 |
| 8 | right_shoulder_pitch_joint.pos | 右臂 |
| 9 | right_shoulder_roll_joint.pos | 右臂 |
| 10 | right_shoulder_yaw_joint.pos | 右臂 |
| 11 | right_elbow_joint.pos | 右臂 |
| 12 | right_wrist_yaw_joint.pos | 右臂 |
| 13 | right_wrist_pitch_joint.pos | 右臂 |
| 14 | right_wrist_roll_joint.pos | 右臂 |
| 15 | right_claw_joint.pos | 右臂 |

**说明**: State 记录各关节的实际位置，Action 是发给各关节的目标位置指令。训练行为克隆模型时，目标是用 State 预测 Action。

## 5. Episode 长度统计

- 最短 episode: 222 帧
- 最长 episode: 354 帧
- 平均长度: 278.96666666666664 帧

**过短 episode (<250帧)**: 6 个 — [13, 14, 16, 17, 23, 24]

**过长 episode (>350帧)**: 1 个 — [8]

![Episode 长度分布](figures/episode_length_barh.png)

## 6. Action / State 统计结果

### 6.1 Action 各维度统计 (全数据集)

| 维度 | Min | Max | Mean | Std |
|-----|-----|-----|------|-----|
| left_shoulder_pitch_joint.pos | -1.2191 | 0.1272 | -0.5046 | 0.3592 |
| left_shoulder_roll_joint.pos | -0.0208 | 0.2549 | 0.0842 | 0.0498 |
| left_shoulder_yaw_joint.pos | -0.3448 | 0.4250 | -0.0619 | 0.1360 |
| left_elbow_joint.pos | -1.6229 | -0.1727 | -0.6911 | 0.3750 |
| left_wrist_yaw_joint.pos | -0.2170 | 0.1842 | 0.0262 | 0.0848 |
| left_wrist_pitch_joint.pos | -0.4666 | -0.1107 | -0.3545 | 0.0829 |
| left_wrist_roll_joint.pos | -0.2986 | 0.1085 | 0.0003 | 0.0712 |
| left_claw_joint.pos | 0.0000 | 0.9800 | 0.5834 | 0.4034 |
| right_shoulder_pitch_joint.pos | -0.0092 | 0.0020 | -0.0027 | 0.0021 |
| right_shoulder_roll_joint.pos | -0.0425 | -0.0330 | -0.0370 | 0.0022 |
| right_shoulder_yaw_joint.pos | -0.0022 | -0.0018 | -0.0019 | 0.0001 |
| right_elbow_joint.pos | -0.0041 | 0.0033 | -0.0022 | 0.0013 |
| right_wrist_yaw_joint.pos | -0.0130 | -0.0129 | -0.0130 | 0.0000 |
| right_wrist_pitch_joint.pos | -0.0008 | 0.0006 | 0.0001 | 0.0001 |
| right_wrist_roll_joint.pos | -0.0364 | -0.0216 | -0.0275 | 0.0029 |
| right_claw_joint.pos | 0.0300 | 0.0300 | 0.0300 | 0.0000 |

![Action 范围](figures/action_ridgeline.png)

![Action 变化量](figures/action_change_line.png)

**低方差 Action 维度** (阈值=0.0001): ['right_shoulder_pitch_joint.pos', 'right_shoulder_roll_joint.pos', 'right_shoulder_yaw_joint.pos', 'right_elbow_joint.pos', 'right_wrist_yaw_joint.pos', 'right_wrist_pitch_joint.pos', 'right_wrist_roll_joint.pos', 'right_claw_joint.pos']
## 7. 数据质量检查结果

详细检查报告见 [check_report.md](check_report.md)

### 检查结论

- 总 episode 数: 30
- 过短 episode: 6 个
- 过长 episode: 1 个
## 8. 可视化图表

![action_change_line.png](figures/action_change_line.png)

![action_ridgeline.png](figures/action_ridgeline.png)

![episode_length_barh.png](figures/episode_length_barh.png)

## 9. 结论与改进建议

### 发现的问题

- 存在 6 个过短 episode，可能需要检查这些 episode 是否未完成任务。
- 存在 1 个过长 episode，可能包含多余动作。
- Action 中有 8 个低方差维度 (['right_shoulder_pitch_joint.pos', 'right_shoulder_roll_joint.pos', 'right_shoulder_yaw_joint.pos', 'right_elbow_joint.pos', 'right_wrist_yaw_joint.pos', 'right_wrist_pitch_joint.pos', 'right_wrist_roll_joint.pos', 'right_claw_joint.pos'])，在训练时可能可以作为常数丢弃。

### 数据集可用性评估

该数据集包含 30 个 episode、8369 帧，涵盖 4 个相机视角。整体数据质量良好，适合用于训练行为克隆模型。建议先清理上述标注异常的 episode 后再用于训练。