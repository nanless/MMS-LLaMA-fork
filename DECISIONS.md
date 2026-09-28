# 复现决策记录

## D1 — MMS-LLaMA 训练语料帧数上限：600（2026-09-25，用户决定）

**决定**：正式 433h 复现使用 `task.max_sample_size=600`（24 s），不再沿用上游默认的 500（20 s）。

**依据**（在 `manifest/433h/train.tsv` 上实测）：

| 上限 | 保留样本 | 训练时数 |
|---|---:|---:|
| 500（上游默认） | 144,067 | 303.9 h |
| **600（本决定）** | **164,572** | **435.7 h** |

- 官方发布清单合计 **436.3 h**，官方 433h checkpoint 标称 **433 hours**；
  用 500 实际只训 303.9 h，比标称少 **30%**。
- 被 500 砍掉的 20,600 条中，**20,505 条落在 501–600 帧**，正是 `--seg-duration 24`
  预处理要产出的满长度片段；用 20 s 上限截断与数据准备自相矛盾。
- 600 保留 **99.8%**（仅丢 249 条 <10 帧与 95 条 >600 帧，合计 0.64 h）。
- `min_sample_size=10` 保持不变。

**保留上游文件**：`src/conf/mms-llama.yaml` 未改动，与上游 `e590937` 逐字节一致。
新profile 为 `src/conf/mms-llama-433h-cap600.yaml`（仅 `max_sample_size: 500 -> 600` 一行差异 + 说明头）。

**运行方式**：

```bash
# 主跑（24 s 口径，435.7 h）
CONFIG_NAME=mms-llama-433h-cap600.yaml scripts/train_local.sh

# 对照组（上游默认，303.9 h）
scripts/train_local.sh
```

**待办**：两个口径都跑一遍，用实测 WER 判定上游默认 500 是否为发布笔误。

## D2 — Vox2 伪标签模型：官方 Whisper large-v3（2026-09-25，用户决定）

正式分段训练标签使用官方 Whisper large-v3 逐段转写。它是对作者未公开 recipe 的
**近似**，不得宣称为 MMS-LLaMA 作者的精确伪标签版本。medium.en 的 600,185 条
clip 级标签保留为候选，不覆盖、不用于正式训练标签。

## D3 — 复现范围：仅 MMS-LLaMA 与 Whisper-Flamingo（2026-09-25，用户决定）

Auto-AVSR 与 MuAViC 仅作为数据准备工具，不作为训练目标。
因此 Auto-AVSR 原生 `lrs3_video_seg16s` + 原生 CSV 路线**不再需要构建**。

## D4 — Q-Former 查询容量由 `max_video_frames` 驱动（2026-09-28）

**背景**：`src/model.py` 原先把 Q-Former 的 query token 数硬编码为
`queries_per_sec * 20`（20 秒），与 `task.max_sample_size` 无关。

**改动**（提交 `ae83434`）：改为
`ceil(queries_per_sec * max_video_frames / 25 * (2 if use_sr_predictor else 1))`，
新增 `model.max_video_frames`（默认 500），并在批次需要的 query 数超过容量时抛
`ValueError`。

**必须同步**：`max_video_frames` 必须与 `task.max_sample_size` 一致。
cap600 profile 已设为 600（提交 `e08bd38`）；否则 24 秒片段会触发新的 `ValueError`。

| profile | max_sample_size | max_video_frames | max_queries |
|---|---:|---:|---:|
| `mms-llama.yaml`（上游） | 500 | 500（默认） | 120 |
| `mms-llama-433h-cap600.yaml` | 600 | **600** | 144 |

## D5 — 多卡失败快速失败：`FAIRSEQ_DIST_TIMEOUT`（2026-09-28）

**背景**：fairseq 在 `init_process_group` 里把集合通信超时硬编码为 **5400 秒**。
实测中一个 rank 死亡后，其余 rank 会在集合通信上阻塞 **90 分钟**，期间每个 rank
仍占着约 30 GB 显存不放；后续任何 run 在加载 checkpoint 时都会 CUDA OOM，
且报错信息是误导性的 `0 bytes already allocated`。

**改动**：`fairseq/fairseq/distributed/utils.py` 的超时改为读环境变量
`FAIRSEQ_DIST_TIMEOUT`，**默认值仍为 5400**（不改变上游行为）。
`scripts/train_local.sh` 将其默认设为 **1800**（30 分钟）。

**运维提示**：多卡训练若出现静默卡死，先查
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`，
清理残留进程树后再重启。

## D6 — `train_local.sh` 支持 smoke 与断点续训（2026-09-28）

原先 `train_local.sh` 把 `optimization.max_update` 硬编码为 30000 且不转发额外参数，
导致 smoke 与续训只能另写一份平行脚本。现增加：

- `MAX_UPDATE`（默认 30000，保持上游配方）
- `WARMUP_UPDATES`（默认 500）
- `RESTORE`（非空时传 `checkpoint.restore_file=`）
- 末尾转发 `"$@"`，可直接追加任意 Hydra 覆盖

新增 `scripts/smoke_local.sh`：两阶段（fresh → resume）冒烟，自带 GPU 排空等待，
覆盖"加载 → 前向/反向 → 验证 → checkpoint 保存 → 恢复"全链路。

**已验证**（2026-09-28，1759h manifest，4 × A800）：
全量扫描 `loaded 819756, 0 unaligned`；phase 1 两步、phase 2 恢复后跑到四步，两阶段 rc=0。
