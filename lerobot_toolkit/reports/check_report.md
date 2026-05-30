# 数据质量检查报告

检查时间: 2026-05-24 21:16:40.838746

| Status | Check | Detail |
|--------|-------|--------|
| ✅ OK | Meta: info.json | 存在 |
| ✅ OK | Meta: tasks.jsonl | 存在 |
| ✅ OK | Meta: episodes.jsonl | 存在 |
| ✅ OK | Meta: episodes_stats.jsonl | 存在 |
| ✅ OK | Episode index uniqueness | 无重复 |
| ✅ OK | Task description | 1 个任务描述完整 |
| ⚠️ WARN | Ep 8 length | 过长 354 帧 (阈值 350) |
| ⚠️ WARN | Ep 13 length | 过短 222 帧 (阈值 250) |
| ⚠️ WARN | Ep 14 action jump | 第 [76, 101, 140] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 14 length | 过短 236 帧 (阈值 250) |
| ⚠️ WARN | Ep 15 action jump | 第 [186, 227] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 16 action jump | 第 [35] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 16 length | 过短 235 帧 (阈值 250) |
| ⚠️ WARN | Ep 17 length | 过短 225 帧 (阈值 250) |
| ⚠️ WARN | Ep 18 action jump | 第 [52] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 23 length | 过短 248 帧 (阈值 250) |
| ⚠️ WARN | Ep 24 length | 过短 236 帧 (阈值 250) |
| ⚠️ WARN | Ep 25 action jump | 第 [263] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 28 action jump | 第 [252] 帧存在较大的 action 跳变 |
| ⚠️ WARN | Ep 29 action jump | 第 [8, 11, 15, 20, 34] 帧存在较大的 action 跳变 |

## 汇总

- ✅ OK: 6 项
- ⚠️ WARN: 14 项
- ❌ ERROR: 0 项
