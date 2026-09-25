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
