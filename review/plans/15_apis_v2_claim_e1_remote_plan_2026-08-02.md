# APIS-V2-CLAIM-E1-REMOTE-20260802：远程 Linux 确认性 E1 执行计划

## 基本信息

- 日期：2026-08-02
- 负责人：远程 Linux `an5bi4acenfa1-0`（执行）；主张规范见本地 `10`/`14`
- 状态：计划中（P0 代码缺口已在分支闭合；本机待路径 remap + 指纹 + 开跑）
- Git commit：`feature/apis-v2-claim-p0` @ `f8df40b`（相对 `main` 含 APIS v2 + claim P0）
- 关联文档：
  - `Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`
  - `review/plans/10_apis_v2_claim_validation_plan_2026-07-30.md`
  - `review/plans/14_apis_v2_claim_execution_plan_2026-07-31.md`（上位执行规范）
  - smoke 记录 `review/records/12_*` / `13_*`（**不得**计入 claim）
  - legacy 同机 `09`（**并行独立，禁止混表**）
- 关联数据文件：计划写入 `outputs/journal/dual_shift_apis_v2/claim/`；hold-out `data/claim/paired_holdout_subjects.json`

---

## 1. 实验指导与依据

- 研究问题：在数据约束下，**源域观测协议上的有界有向残差干预（APIS v2）** 是否提升对 acquisition-/cohort-protocol shift 的鲁棒性，且不损害诊断判别。
- 假设与基线（主比较，不可替代）：
  - `apis_v2` vs **`mixstyle`**
  - `apis_v2` vs **`metadata_xda`（X+D+A）**
  - `metadata`（X+A）、`ce_only` 为补充/弱对照；不得单独用来宣称 claim
- 实施依据：分支已实现 residual APIS、hold-out 排除（rev.2）、`balanced_accuracy`、公平 `metadata_xda`、E1 启动器与汇总脚本；`14` 将 P0 检查单标为代码侧已勾选，仅余开跑前指纹。
- 实验范围（本阶段仅 **E1 第一波**）：
  - 任务：CN vs AD
  - 方向：ADNI→NACC、NACC→ADNI（反向须报 ADNI 测试 1.5T/3T 分层）
  - Seeds（训练）：`42,43,44,45,46`；**划分**冻结 `split_seed=42`
  - 变体：`ce_only`, `mixstyle`, `metadata`, `metadata_xda`, `apis_v2`
  - 预算：`epochs=50`，`alpha_max=0.25`，`protocol_revision=2`
- 判定标准（运行前锁定）：
  - **主终点**：subject-level **balanced_accuracy**
  - **Go**：两方向上，跨预注册种子，`Δ(apis_v2−mixstyle)` 与 `Δ(apis_v2−metadata_xda)` 均为正，且各自 95% subject-level 配对 bootstrap CI **不含 0**
  - 次要：AUC / SEN / SPE / macro-F1 / worst-class recall —— 如实报告，不得覆盖主终点失败
  - Checkpoint：仅 source validation；禁止 target 选模/调参
  - 停止：任一方向主终点失败 → 不宣称 v2 性能 claim；可写机制/负结果，不扩任务

---

## 2. 可复现记录

### 2.1 代码与入口（本分支新能力）

| 组件 | 路径 | 作用 |
|---|---|---|
| 冻结配置 | `journal_dual_shift_apis_v2_claim.yaml` | E1 全预算；rev.2；变体清单 |
| 启动器 | `experiments/run_apis_v2_claim_e1.py` | 指纹 + 队列；`max_workers` **硬顶 2**；默认 `--force`（rev.2） |
| 汇总 | `experiments/report_apis_v2_claim_e1.py` | 跨 seed Δ + 配对 bootstrap；拒绝 smoke 路径 |
| Hold-out | `data/claim/paired_holdout_subjects.json` + `utils/claim_holdout.py` | ≤30d **33** subjects 训练前排除 |
| 模型 | `Model/dual_shift/apis.py` 等 | 有向残差 APIS v2；per-sample mask |
| 公平基线 | `Model/dual_shift/metadata_baseline.py` | `metadata` / `metadata_xda` |
| 单测 | `tests/test_apis_v2.py`, `test_claim_holdout.py`, `test_claim_launcher.py` | 组件/协议门控 |

### 2.2 配置与数据（远程 remap）

- 冻结源：`journal_dual_shift_apis_v2_claim.yaml`（超参不得改）
- 本机执行副本（路径-only）：`journal_dual_shift_apis_v2_claim_remote.yaml`
  - `scan_manifest.root` → `./scan_manifests`（或本机等价）
  - `cohorts.*.image_root` / `metadata_csv` → 本机 `./dataset/...`（与 postfix remote 一致）
  - **禁止**改 `epochs`、`alpha_max`、`lambda_*`、`variants`、`claim.*`、`study.seeds`
- Hold-out：`subjects_le_30d`（33 人）；全部扫描不得进 train/val/选模；E3 另册
- Manifest SHA（本机已知 postfix 对齐清单，开跑前写入指纹）：
  - ADNI / NACC scan_manifest 以指纹文件为准

### 2.3 作业矩阵

启动器粒度：**1 job = 1 seed × 1 direction × 5 变体（串行）**  
共 **10 jobs**；并行 ≤ **2**。

| 波次 | Jobs | 说明 |
|---|---|---|
| P1 | 10 × `{42..46} × {adni_to_nacc, nacc_to_adni}` | E1 第一波（必须） |
| P2（后） | `film` / `legacy_apis` / `uncond_residual` | claim 完整性；本计划不自动开 |
| P3（后） | `apis_v2_shuffle` | 机制负对照 |
| E3 | 配对 ≤30d 评估 | 用未见 hold-out 的 ADNI 源 ckpt；不替代 E1 |

### 2.4 环境与命令（预注册）

- Host：`an5bi4acenfa1-0`；env `cyh`；PyTorch 2.13+cu130；6× RTX 5090  
- **注意**：claim 启动器硬顶 `max_workers=2`（勿改为 3；与 legacy `apis_3seed` 队列隔离）
- 与 `09` samehost / 旧 `dual_shift_apis_3seed` **输出根隔离**，禁止混均值

```bash
cd /path/to/dual_shift_github   # feature/apis-v2-claim-p0 @ f8df40b
export JOURNAL_PYTHON=/opt/conda/envs/cyh/bin/python
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

# T0：单测 + 指纹（进入条件）
$JOURNAL_PYTHON -m unittest discover -s tests -p 'test_apis*.py' -v
$JOURNAL_PYTHON -m unittest discover -s tests -p 'test_claim*.py' -v
$JOURNAL_PYTHON experiments/run_apis_v2_claim_e1.py \
  --config_path journal_dual_shift_apis_v2_claim_remote.yaml \
  --fingerprint-only

# T1：E1 第一波（默认 force=rev2；确认后去掉误跑的 --no-force）
$JOURNAL_PYTHON experiments/run_apis_v2_claim_e1.py \
  --device cuda --max-workers 2 \
  --config_path journal_dual_shift_apis_v2_claim_remote.yaml \
  --seeds 42,43,44,45,46

# T2：汇总（全部 job ok 后）
$JOURNAL_PYTHON experiments/report_apis_v2_claim_e1.py \
  --seed-root outputs/journal/dual_shift_apis_v2/claim/e1 \
  --seeds 42,43,44,45,46 \
  --output-dir outputs/journal/dual_shift_apis_v2/claim/e1
```

### 2.5 墙钟粗估（本机 5090，参考 legacy 3 变体 ~7–10 h/方向）

- 单 job（5 变体串行）≈ **12–17 h**
- 10 jobs × 2 并行 ≈ **5 波** → 墙钟约 **60–85 h（约 2.5–3.5 天）**
- 若 GPU 被其他任务占用则顺延；不得抢杀他人进程

### 2.6 产物

```text
outputs/journal/dual_shift_apis_v2/claim/
  env_fingerprint.json
  protocol_freeze/journal_dual_shift_apis_v2_claim_remote.yaml
  e1/seed{S}/{adni_to_nacc|nacc_to_adni}/{variant}/
    journal_metrics.json  # 含 balanced_accuracy, claim_protocol_revision, split_seed…
    split_manifest.json   # hold-out 交集断言
    target_predictions.csv
  e1/gate_report_claim.json
  e1/metrics_table_claim.csv
```

---

## 3. 分析与结果（预留）

### 3.1 结果

| 方法/条件 | balanced_acc（主） | AUC / F1 / SEN / SPE | 判定 | 备注 |
|---|---:|---:|---|---|
| ce_only | 待跑 | 待跑 | 弱基线 | |
| mixstyle | 待跑 | 待跑 | 主对照 1 | |
| metadata | 待跑 | 待跑 | X+A 补充 | |
| metadata_xda | 待跑 | 待跑 | 主对照 2 | |
| apis_v2 | 待跑 | 待跑 | 方法 | |
| Δ(v2−mixstyle) + CI | 待跑 | — | 主判定 | |
| Δ(v2−metadata_xda) + CI | 待跑 | — | 主判定 | |

### 3.2 分析

- 相对基线：仅同机同协议 rev.2 结果；**禁止**与 smoke、legacy AdaIN 3-seed、Windows 表混算
- 异常与局限：NACC 无 1.5T；配对非同步；场强与厂家/序列纠缠；不得写成场强因果
- 结果结论：仅当 §1 Go 满足方可写性能主结论

---

## 4. 建议下一步实验指导

- 建议动作（本节点）：
  1. 检出 `feature/apis-v2-claim-p0`，写 `_remote.yaml` 路径 remap  
  2. T0 单测 + fingerprint  
  3. 确认无冲突 GPU 任务后启动 T1  
  4. T2 汇总 → 新开 `review/analysis/` 文档判定 Go/No-Go  
- 建议依据：`14` P0 代码已闭合；确认性矩阵尚未在任何主机以完整预算跑完  
- 固定条件：`protocol_revision=2`、`split_seed=42`、`alpha_max=0.25`、5 变体、5 训练种子、hold-out ≤30d、source-only 选模、主终点 balanced_accuracy  
- 进入条件：单测绿；指纹与 `protocol_freeze/` 已写；manifest/图像路径可读；claim 输出根无误用 smoke  
- 禁止事项：
  - 用 6-epoch smoke 报 claim  
  - 混用 legacy Gate A / `09` samehost 结果  
  - 扩 MCI / 三分类 / Joint / CDT 多 seed  
  - 看 target 后改 APIS 强度、warmup、距离或损失权重  
  - 把 NACC→ADNI 改善写成“场强因果机制已证明”  
  - 将 `max_workers` 提到硬顶 2 以上绕过启动器（若改代码须先改协议文档）

### 开跑前检查单

- [ ] 工作树 = `feature/apis-v2-claim-p0` @ `f8df40b`（或更新后记录新 HEAD）
- [ ] `_remote.yaml` 仅路径差异；`claim.protocol_revision == 2`
- [ ] `subjects_le_30d` 可读，计数 33
- [ ] unittest（apis + claim）通过
- [ ] `env_fingerprint.json` 已写
- [ ] 与 `dual_shift_apis_3seed*` / samehost 输出隔离
- [ ] GPU 空闲策略确认（≤2 卡给 claim）

---

## 附录 A：与旧实验关系

| 产物 | 角色 |
|---|---|
| Windows legacy APIS 3-seed | Gate A **No-Go**；不同主张 |
| Linux `09` samehost | AdaIN 环境复现；独立 |
| `smoke/` 6-epoch | 通路 only |
| 本计划 `claim/e1/` | **唯一** v2 性能确认性汇总根 |

## 附录 B：输入轴（冻结）

| 变体 | X | D | A | 备注 |
|---|---:|---:|---:|---|
| ce_only | ✓ | | | |
| mixstyle | ✓ | ✓ | | MixStyle + 人口学融合 |
| metadata | ✓ | | ✓ | 补充 A-only |
| metadata_xda | ✓ | ✓ | ✓ | 公平主对照 |
| apis_v2 | ✓ | ✓ | 训练期 | 残差干预；推理走 clean |
