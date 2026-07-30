# Dual-Shift 远程实验交接与任务分发

**创建日期：** 2026-07-29  
**文档序号：** `04`  
**继承：**  
- `review/plans/01_dual_shift_next_step_decision_2026-07-29.md`（决策边界，生效）
- `review/plans/02_dual_shift_experiment_schedule_2026-07-29.md`（执行安排）
- `review/records/03_dual_shift_apis_3seed_status_2026-07-29.md`（本机进度快照）
**用途：** 项目已复制到远程服务器后，按本文件分发、认领与验收实验任务。  

---

## 0. 一分钟摘要

当前正式证据只认 **postfix / CN vs AD / seed 42**。下一步唯一主实验是：用**冻结**的 `config/journal_dual_shift_postfix.yaml` 补齐 **APIS 双向 3-seed（43/44）**，变体仅 `ce_only`、`mixstyle`、`apis_only`。完成后汇总 Gate A；**通过前禁止扩 MCI / Joint / CDT 多 seed / 调超参**。

| 工作包 | 谁做 | GPU | 状态（交接时） |
|---|---|---|---|
| P0 协议冻结 | 已完成，远程只需核对 | 否 | 完成 |
| P1 APIS 3-seed（43/44） | **主分发任务** | 是 | 本机队列可能已部分跑；远程需防重复 |
| P1b 配对 1.5T/3T | 已完成；远程可选复跑/补覆盖 | 轻量 | 完成（21/34 对） |
| P2 CDT source 审计 | 已完成 | 否 | 完成，非病态 |
| P3 Gate A 汇总 | 全部 P1 结束后一人汇总 | 否 | 待做 |

---

## 1. 硬约束（所有节点必须遵守）

### 1.1 允许

- 用冻结 postfix 配置跑 `ce_only` / `mixstyle` / `apis_only`
- 方向：`ADNI_to_NACC`、`NACC_to_ADNI`
- Seeds：`43`、`44`（`42` 复用 postfix，不重训除非实现 bug）
- 训练失败按技术原因重跑并留日志
- 机制评估、日志、审计（不改模型输出）

### 1.2 禁止

- 扩 MCI vs AD / MCI vs CN / 三分类
- 看 target 后调 APIS / CDT / checkpoint / warmup / 损失权重
- 跑 `cdt_only` / `dual_shift` 的 seed 43/44
- Joint 补跑或调参
- 把 `dual_shift_stage2/` 与 postfix 混表
- 改冻结配置却不改文档序号并全员通知

### 1.3 正式证据口径

- 正式结果：`outputs/journal/dual_shift_postfix/`（seed 42）+ `outputs/journal/dual_shift_apis_3seed/`（43/44）
- Stage2：`outputs/journal/dual_shift_stage2/` = 历史探索，不进正式表
- 训练日志里的 `val_auc` 可能是 **composite score**（可 >1），论文/gate 以 `journal_metrics.json` 为准

---

## 2. 远程环境与路径改写（先做）

### 2.1 代码与 Python

```bash
cd <REPO_ROOT>
export PYTHONPATH=<REPO_ROOT>
# 建议使用与本机一致的 CUDA 环境；示例：
# python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

关键入口：

| 脚本 | 作用 |
|---|---|
| `scripts/run_journal_queue.py --stages apis_3seed` | 一键 4 个方向×seed 任务 |
| `run_v2.py --exp journal ...` | 单 job 手工分片 |
| `scripts/report_apis_3seed.py` | 3-seed 汇总 + Gate A 预览 |
| `scripts/run_paired_field_strength_batch.py` | 配对机制批评估 |
| `scripts/audit_cdt_source.py` | CDT source 审计（已完成可跳过） |

配置（**不得改超参**）：`config/journal_dual_shift_postfix.yaml`

### 2.2 数据路径（本机 → 远程必须对齐）

本机冻结配置中的路径：

| 键 | 本机路径 |
|---|---|
| `scan_manifest.root` | `F:/ADNI/scan_manifests` |
| ADNI `image_root` | `F:/ADNI/ADNI_dataset` |
| NACC `image_root` | `F:/ADNI/NACC_dataset` |
| ADNI `metadata_csv` | `D:/ADNI/merge.csv` |
| NACC `metadata_csv` | `F:/NACC/MRI_mulclass3.csv` |
| 配对表 | `F:/ADNI/scan_manifests/paired_field_strength_manifest.csv` |

**远程做法（二选一，全员统一）：**

1. **推荐**：在服务器上挂载/软链到相同逻辑路径；或  
2. 复制一份 config 为 `config/journal_dual_shift_postfix_remote.yaml`，**只改路径字段**，其余逐字节保持一致，并在任务认领表中写明所用 config。

改路径后先做烟测（可选，CPU/短跑）：

```bash
python scripts/run_journal_queue.py --stages smoke_dual_shift --device cuda --max-workers 1 --force
```

### 2.3 必须随仓库或单独同步的产物

远程若要做汇总或跳过重训，至少需要：

```text
outputs/journal/dual_shift_postfix/
  adni_to_nacc/{ce_only,mixstyle,apis_only,cdt_only}/journal_metrics.json (+ best_checkpoint.pt 若要配对)
  nacc_to_adni/...
  protocol_freeze_2026-07-29/          # P0 冻结档案
outputs/journal/dual_shift_apis_3seed/
  cdt_source_audit_seed42.json         # 可选
  paired_field_strength_seed42.json    # 可选；也可远程复跑
```

`best_checkpoint.pt` 体积大：若远程只训 43/44、本机已有 42，汇总时把 postfix metrics 拷回即可。

---

## 3. 防重复：本机与远程协调

交接时本机已启动：

```bash
python scripts/run_journal_queue.py --stages apis_3seed --device cuda --max-workers 1 --force
```

日志目录示例：`outputs/journal/queue_logs/20260729_120345/`  
进度快照：`seed43/adni_to_nacc` 已进入 `mixstyle`（约 epoch 20+）。

**规则：**

1. **同一 `(seed, direction, variant)` 只允许一个节点写入。**  
2. 远程启动前先检查是否已有：
   - `outputs/journal/dual_shift_apis_3seed/seed{S}/{dir}/{variant}/journal_metrics.json`
   - 或本机队列仍在跑同一输出目录  
3. 多机分片时：**不要**对已在跑的目录再加 `--force` / `--force-variants`。  
4. 推荐协调方式（选一）：
   - **A. 停本机，全部改远程串行队列**（最简单）  
   - **B. 按 seed 分机**：节点 A=`seed43`，节点 B=`seed44`（见 §4.2）  
   - **C. 按方向分机**：节点 A=`ADNI_to_NACC`，节点 B=`NACC_to_ADNI`（每节点仍含 seed 43+44）

---

## 4. 可分发任务包

### 任务包 T0 — 环境验收（每台远程先做）

**负责人：** 各节点  
**验收：**

- [ ] CUDA 可用，`batch_size=4` 不 OOM  
- [ ] ADNI/NACC 图像与 manifest 可读；随机抽 1 个 nii 可加载  
- [ ] 确认使用的 config 路径字段正确，且除路径外与冻结 postfix 一致  
- [ ] 已与其他节点约定分片方案（A/B/C）

---

### 任务包 T1 — APIS 3-seed 主训练（GPU，最高优先）

**目标输出：**

```text
outputs/journal/dual_shift_apis_3seed/
  seed43/adni_to_nacc/{ce_only,mixstyle,apis_only}/
  seed43/nacc_to_adni/{ce_only,mixstyle,apis_only}/
  seed44/adni_to_nacc/...
  seed44/nacc_to_adni/...
```

每个变体目录验收文件：

- `best_checkpoint.pt`
- `journal_metrics.json`
- 父目录最终有 `summary.json`（队列跑完时）

**预计：** 单 GPU 串行约 **12–18h**（2 seed × 2 方向 × 3 变体 × ~1–1.5h）。

#### 4.1 单机整包（方案 A）

```bash
cd <REPO_ROOT>
export PYTHONPATH=<REPO_ROOT>
python scripts/run_journal_queue.py \
  --stages apis_3seed \
  --device cuda \
  --max-workers 1 \
  --force
```

> 若本机已跑完部分变体且目录已同步到远程：去掉 `--force`，队列会跳过已有 `summary.json` 的 job；**变体级**跳过需人工确认，必要时按 §4.2 只补缺失 job。

#### 4.2 多机分片示例（方案 B：按 seed）

**节点 A — seed 43**

```bash
# ADNI → NACC
python run_v2.py --exp journal --direction ADNI_to_NACC \
  --variants ce_only mixstyle apis_only --seed 43 --device cuda \
  --output-dir outputs/journal/dual_shift_apis_3seed/seed43/adni_to_nacc \
  --config_path config/journal_dual_shift_postfix.yaml

# NACC → ADNI
python run_v2.py --exp journal --direction NACC_to_ADNI \
  --variants ce_only mixstyle apis_only --seed 43 --device cuda \
  --output-dir outputs/journal/dual_shift_apis_3seed/seed43/nacc_to_adni \
  --config_path config/journal_dual_shift_postfix.yaml
```

**节点 B — seed 44**（同上，改 `seed 44` 与输出路径 `seed44/...`）

远程若改了路径 config，把 `--config_path` 换成 `config/journal_dual_shift_postfix_remote.yaml`。

#### 4.3 任务认领表（复制到共享表格）

| Job ID | seed | direction | variants | 节点 | 开始 | 结束 | 产物路径 | 备注 |
|---|---:|---|---|---|---|---|---|---|
| J43A | 43 | ADNI_to_NACC | ce/mix/apis | | | | `.../seed43/adni_to_nacc/` | |
| J43N | 43 | NACC_to_ADNI | ce/mix/apis | | | | `.../seed43/nacc_to_adni/` | |
| J44A | 44 | ADNI_to_NACC | ce/mix/apis | | | | `.../seed44/adni_to_nacc/` | |
| J44N | 44 | NACC_to_ADNI | ce/mix/apis | | | | `.../seed44/nacc_to_adni/` | |

---

### 任务包 T2 — 配对机制（可选复跑 / 补覆盖）

本机已完成：`outputs/journal/dual_shift_apis_3seed/paired_field_strength_seed42.json`  
结论摘要：≤90d 可评估 **21/34** 对；`apis_only` mean\|Δp\| 低于 CE/MixStyle；flip=0；无系统性反证。

远程仅在下列情况复跑：

- 想提高 21→34 覆盖（先修 manifest/图像路径对齐）  
- 或本机 JSON 未同步

```bash
python scripts/run_paired_field_strength_batch.py \
  --config config/journal_dual_shift_postfix.yaml \
  --pairs <PAIRED_CSV> \
  --max-days 90 \
  --device cuda \
  --ckpt-root outputs/journal/dual_shift_postfix/adni_to_nacc \
  --output outputs/journal/dual_shift_apis_3seed/paired_field_strength_seed42.json
```

---

### 任务包 T3 — CDT 审计（默认跳过）

已完成且 **非 pathological**，**不触发**一次稳定化修复。  
除非怀疑 checkpoint 未同步或审计脚本更新，否则远程不要重做，更不要据此调 CDT。

---

### 任务包 T4 — Gate A 汇总（P1 全部完成后，一人执行）

**前置：** 四个 job 的三变体 `journal_metrics.json` 齐全；postfix seed42 指标可读。

```bash
python scripts/report_apis_3seed.py \
  --postfix-root outputs/journal/dual_shift_postfix \
  --seed-root outputs/journal/dual_shift_apis_3seed \
  --seeds 42,43,44 \
  --output-dir outputs/journal/dual_shift_apis_3seed
```

**Gate A（进入 MCI 的最低条件，缺一不可）：**

1. 双方向 × 3 seeds 完成，无未解释训练/checkpoint 异常  
2. **两方向**均：AUC 相对 CE 非劣（差 ≤0.01）、F1 提升、无 SEN/SPE 塌缩  
3. 非单 seed 驱动；配对机制无系统性反证  
4. MixStyle 完整报告（不作事后 gate）

| 结果 | 动作 |
|---|---|
| **Go** | 另写 `05_...` 启动**一个**预注册 MCI 先导（MCI vs AD **或** MCI vs CN） |
| **No-Go** | 停扩任务；写机制/负结果；论文收缩为方向依赖 + 机制边界 |
| APIS 过门但 CDT 仍 fail | 主线 → protocol-aware DG；Joint 不作主贡献 |

汇总后更新：`03_...` 或新建 `05_dual_shift_gate_a_decision_YYYY-MM-DD.md`。

---

## 5. 交付物清单（汇交负责人）

训练节点交付：

- [ ] 各 job 的 `journal_metrics.json` + `best_checkpoint.pt`（或约定只交 metrics）  
- [ ] 队列/训练日志（`outputs/journal/queue_logs/<stamp>/` 或 job stdout）  
- [ ] 使用的 config 路径说明（若 remote 改路径）  
- [ ] 任务认领表填完

汇总节点交付：

- [ ] `gate_report_3seed.json` / `metrics_table_3seed.csv`（或 `report_apis_3seed.py` 实际写出的文件）  
- [ ] Gate A **Go / No-Go** 书面结论（引用冻结 gate，禁止事后改规则）  
- [ ] 配对结果引用或复跑 JSON  

---

## 6. 明确不启动清单（远程同样禁止）

- `cdt_only` / `dual_shift` 的 seed 43/44  
- Joint 任何补跑  
- MCI / 三分类  
- 查看 target 后改 APIS 强度、warmup、距离、损失权重或 CDT 温度/ESS  
- 混用 Stage2 与 postfix 数字  

---

## 7. 推荐分发顺序

```text
1) 全员读 01 + 本交接文档；填认领表；统一路径方案
2) 各节点完成 T0
3) 协调停本机或按 seed/方向分片 → 执行 T1
4) （可选）T2 补配对覆盖
5) 汇交 metrics → T4 Gate A
6) 仅 Gate A Go 后另开文档做 MCI；否则收束论文叙事
```

---

## 8. 联系与文档索引

| 文档 | 角色 |
|---|---|
| `review/plans/01_dual_shift_next_step_decision_2026-07-29.md` | 决策与不可宣称边界 |
| `review/plans/02_dual_shift_experiment_schedule_2026-07-29.md` | 本机执行安排 |
| `review/records/03_dual_shift_apis_3seed_status_2026-07-29.md` | 本机进度与 CDT/配对结论 |
| **本文 `04_...`** | **远程分发与认领** |

问题升级顺序：路径/数据 → 训练崩溃日志 → 是否违反冻结配置 → 是否与本机重复写同一目录。
