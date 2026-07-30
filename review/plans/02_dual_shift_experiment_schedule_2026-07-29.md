# Dual-Shift 后续实验执行安排

**创建日期：** 2026-07-29  
**文档序号：** `02`  
**继承决策：** `review/plans/01_dual_shift_next_step_decision_2026-07-29.md`
**状态：** 生效 / 可执行  

---

## 0. 决策摘要（约束）

| 允许 | 禁止 |
|---|---|
| 冻结 postfix 后补 APIS 3-seed（43/44） | 扩 MCI / 三分类 |
| ADNI 同受试者 1.5T/3T 配对机制验证 | 为过 target gate 调 APIS/CDT |
| CDT **source-side** 审计（必要时一次稳定化） | CDT/Joint 性能扩 seed |
| MixStyle 完整报告作强对照 | Joint 调参或补跑 |

**变体范围：** 仅 `ce_only`（=ds_ce）、`mixstyle`、`apis_only`  
**方向：** ADNI→NACC、NACC→ADNI  
**Seeds：** 42（已有 postfix）+ 43 + 44  

---

## 1. 工作包与优先级

```text
P0  冻结 postfix 协议档案
P1  APIS 双向 3-seed 补齐（seed 43/44）     ← 主队列（GPU）
P1b ADNI 1.5T/3T 配对机制验证（seed42 ckpt） ← 可与 P1 并行（评估）
P2  CDT source-side 审计（seed42 cdt_only）  ← CPU/轻量
P3  Gate A 汇总 → Go/No-Go（不开 MCI）
```

预计 GPU 墙钟：约 **2 方向 × 2 seed × 3 变体 × ~1–1.5h/变体 ≈ 12–18h**（串行 max-workers=1）。

---

## 2. P0 — 冻结协议档案

**目录：** `outputs/journal/dual_shift_postfix/protocol_freeze_2026-07-29/`

留档内容：

1. `config/journal_dual_shift_postfix.yaml` 副本  
2. `split_manifest.json`（两方向）  
3. `gate_report.json` + `stage2_metrics_table.csv`  
4. 关键文件指纹：`Model/dual_shift/*.py`、`training/dual_shift_loop.py`、`data/journal_dataset.py`、`data/scan_manifest.py`  
5. 冻结规则摘要 JSON：checkpoint 公式、collapse guard、APIS/MixStyle 超参、评估聚合方式  

**验收：** 指纹文件存在；后续 seed 43/44 使用同一 config path，不改超参。

---

## 3. P1 — APIS 3-seed 补齐（主实验）

### 3.1 输出布局

```text
outputs/journal/dual_shift_apis_3seed/
  seed42/   → 符号链接或清单指向 dual_shift_postfix/{adni_to_nacc,nacc_to_adni}
              （仅 ce_only, mixstyle, apis_only 三变体计入汇总）
  seed43/
    adni_to_nacc/{ce_only,mixstyle,apis_only}/
    nacc_to_adni/{ce_only,mixstyle,apis_only}/
  seed44/
    adni_to_nacc/...
    nacc_to_adni/...
  gate_report_3seed.json
  metrics_table_3seed.csv
```

### 3.2 队列命令

```bash
# 仅补 43/44；配置与 postfix 完全一致
python scripts/run_journal_queue.py --stages apis_3seed --device cuda --max-workers 1 --force
```

队列行为（已实现）：

- variants = `ce_only mixstyle apis_only`（不含 cdt_only / dual_shift）  
- seeds = 43, 44  
- config = `config/journal_dual_shift_postfix.yaml`  
- output = `outputs/journal/dual_shift_apis_3seed/seed{S}/{direction}/`  

### 3.3 完成后汇总

```bash
python scripts/report_apis_3seed.py \
  --postfix-root outputs/journal/dual_shift_postfix \
  --seed-root outputs/journal/dual_shift_apis_3seed \
  --seeds 42,43,44 \
  --output-dir outputs/journal/dual_shift_apis_3seed
```

报告必须包含：逐 seed 指标、跨 seed mean±std、`apis−ce` / `apis−mixstyle` 差值、逐 seed APIS gate、方向级稳定性结论。

### 3.4 Gate A（进入 MCI 的最低条件）— 复述

- 双方向 × 3 seeds 完成且无未解释训练/checkpoint 异常  
- **两方向**均满足：AUC 非劣（≤0.01）、F1 提升、无 SEN/SPE 塌缩  
- 非单 seed 驱动；配对机制不出现系统性反证  
- MixStyle 如实报告（不作事后 gate）

**失败则：** 停扩任务；论文收缩为方向依赖探索 + 机制边界。

---

## 4. P1b — 配对 1.5T/3T 机制验证（并行）

### 4.1 构建配对表

```bash
python scripts/build_paired_field_strength_manifest.py \
  --manifest F:/ADNI/scan_manifests/ADNI_scan_manifest.csv \
  --output F:/ADNI/scan_manifests/paired_field_strength_manifest.csv
```

实际产物：`F:/ADNI/scan_manifests/paired_field_strength_manifest.csv`（及 summary JSON）。  
报告：配对数、day_gap 分布、label_match 率、≤30/90/180d 计数。

### 4.2 评估（seed 42 postfix checkpoints）

对 `ce_only`、`mixstyle`、`apis_only` × ADNI→NACC 的三个 checkpoint（主分析）；NACC→ADNI 可作敏感性：

```bash
python scripts/run_paired_field_strength_batch.py \
  --config config/journal_dual_shift_postfix.yaml \
  --pairs F:/ADNI/scan_manifests/paired_field_strength_manifest.csv \
  --max-days 90 \
  --device cpu \
  --output outputs/journal/dual_shift_apis_3seed/paired_field_strength_seed42.json
```

### 4.3 必报指标（对齐决策 §8.2）

1. \(|p_{1.5}-p_{3}|\) 均值/中位/分位 + bootstrap CI  
2. 有符号 logit 差  
3. 类别翻转率  
4. Spearman/Pearson  
5. 表征距离 vs 非配对对照  
6. 场强 probe（可选，若脚本已支持则跑）  
7. 1.5T/3T 子集 AUC/F1 分层  
8. 逐 pair APIS−CE / APIS−MixStyle 改善  

---

## 5. P2 — CDT source-side 审计（不扩 seed）

```bash
python scripts/audit_cdt_source.py \
  --run-dirs \
    outputs/journal/dual_shift_postfix/adni_to_nacc/cdt_only \
    outputs/journal/dual_shift_postfix/nacc_to_adni/cdt_only \
  --output outputs/journal/dual_shift_apis_3seed/cdt_source_audit_seed42.json
```

审计项：subject-level 风险、ESS、权重集中度、风险曲面摘要、missing 支持、NaN/极端权重。  

**规则：** 仅当 source 出现预定义数值病态时，允许**一次**稳定化修复并完整报告；target gate 失败本身不触发修复。

---

## 6. P3 — Gate A 决策节点

| 结果 | 动作 |
|---|---|
| Gate A **Go** | 预注册选定 **一个** MCI 先导（MCI vs AD **或** MCI vs CN），再写 `03_...` 启动文档 |
| Gate A **No-Go** | 停扩；写机制/负结果文档；论文转 protocol-aware 探索或进一步收缩 |
| CDT 审计健康但性能 fail | CDT 降为审计/负结果；不恢复 Joint |
| APIS 过门 + CDT fail | 论文主线 → **protocol-aware DG**；Joint 不作主贡献 |

---

## 7. 明确不启动清单

- `cdt_only` / `dual_shift` 的 seed 43/44  
- Joint 任何形式补跑或调参  
- MCI vs AD / MCI vs CN / 三分类  
- 查看 target 后改 APIS 强度、warmup、距离、损失权重或 CDT 温度/ESS  

---

## 8. 立即执行清单（本会话）

- [x] 撰写本安排文档  
- [x] 创建 protocol freeze 档案  
- [x] 实现/接入 `apis_3seed` 队列阶段  
- [x] 启动 seed 43/44 GPU 队列  
- [x] 构建配对表并启动 seed42 配对评估  
- [x] 落地 CDT 审计脚本并跑 seed42  

进度与 Gate A 预览见：`review/records/03_dual_shift_apis_3seed_status_2026-07-29.md`。
