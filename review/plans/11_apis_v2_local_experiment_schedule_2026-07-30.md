# APIS-V2-LOCAL-20260730：本地 APIS v2 就绪与首轮验证安排

> **校正（2026-07-31）：** 本文保留为「就绪门控 + smoke 通路」历史安排。  
> **v2 主张验证的正式执行计划**请改用  
> [`14_apis_v2_claim_execution_plan_2026-07-31.md`](14_apis_v2_claim_execution_plan_2026-07-31.md)。  
> 本文中将短程 smoke 数值、legacy Gate A、与完整 E1 混写的部分 **不再作为 claim 依据**。

## 基本信息

- 日期：2026-07-30
- 负责人：本地 Windows 工作区（`merge/apis-v2-local`）
- 状态：计划中
- Git commit：合并后基线 `1c50fd6`（`origin/feature/apis-v2` @ `35aa44c` + 本地 weights 历史）
- 关联文档：`review/plans/10_apis_v2_claim_validation_plan_2026-07-30.md`、`Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`、`review/analysis/08_dual_shift_apis_3seed_analysis_2026-07-30.md`、`review/plans/09_apis_3seed_samehost_repro_plan_2026-07-30.md`、`docs/EXPERIMENT_RECORD_TEMPLATE.md`
- 关联数据文件：计划写入 `outputs/journal/dual_shift_apis_v2/`；就绪门控前不启动完整矩阵

## 1. 实验指导与依据

- 研究问题：在合并 `feature/apis-v2` 后，本地能否按数据约束设计完成 APIS v2 端到端就绪，并启动可辩护的首轮外部队列泛化（E1）验证。
- 假设与基线：APIS v2（观测协议残差干预）相对 `ce_only`、`mixstyle` 与直接 metadata 条件化，在 subject-level balanced accuracy 上具有可复现增益；机制证据不替代性能主终点。
- 实施依据：
  - 正式 Gate A（legacy AdaIN APIS，Windows 3-seed）= **No-Go**（`review/analysis/05_...` / `08_...`）。
  - Windows/Linux 结果不可混用；同机复现（`09`）面向预注册 Linux 参考主机，**不作为本机 Windows 立即全量重跑任务**。
  - 本分支新增主张与实验矩阵以 `10` + `APIS_V2_DATA_CONSTRAINED_DESIGN.md` 为准。
- 实验范围（本地本阶段）：
  - 任务：CN vs AD
  - 方向：先 ADNI→NACC smoke，再双向 E1
  - Seeds：smoke=`42`；确认性至少 `42,43,44`（完整确认性按 `10` 要求扩至 ≥5）
  - 方法：`ce_only`、`mixstyle`、legacy APIS（对照）、`apis_v2`；metadata/FiLM 等在就绪后补齐
- 判定标准：
  - 就绪门控（§2）全部通过后才允许共享完整矩阵
  - E1 主终点：subject-level balanced accuracy；相对 MixStyle **与** metadata 条件化的差值，95% 配对 bootstrap CI 不含 0
  - 停止条件：实现错误可修；**禁止**查看 target 后改 APIS/CDT 超参；不因 legacy Gate A No-Go 而扩 MCI/Joint

## 2. 可复现记录

- 配置文件：`config/journal_dual_shift_postfix.yaml`（超参冻结模板）；APIS v2 专用字段待就绪后新增 `config/journal_dual_shift_apis_v2.yaml`（仅声明 v2 开关与对照，不改 postfix 超参语义）
- 数据与划分版本：沿用既有 ADNI/NACC scan manifest；配对机制集（≤30d / ≤7d）受试者必须整主体排除训练与模型选择
- 随机种子：smoke `42`；随后 `43,44`（再扩 45/46）
- 环境与硬件：本地 Windows；Python 建议 `C:\Anaconda3\envs\pytorch`；启动前写 `outputs/journal/dual_shift_apis_v2/env_fingerprint.json`
- 启动命令：见下方工作包
- 工作区状态：dirty（本地 weights/logs 与 `_local_pre_merge_backup/`）；实验相关代码以合并后的 `Model/dual_shift/*`（v2）+ 恢复的 `training/`、`scripts/` 为准
- 产物位置：`outputs/journal/dual_shift_apis_v2/`；legacy 结果保留在 `outputs/apis_3seed/`、`outputs/journal/dual_shift_apis_3seed/`、`outputs/journal/dual_shift_postfix/`

### 2.1 就绪门控现状（对照 `10` §9）

| # | 门控 | 本地状态 | 动作 |
|---|---|---|---|
| R1 | `training/dual_shift_loop.py` 可用 | **已恢复** | 保持；纳入版本库 |
| R2 | `Model/__init__.py` 遗留导入不阻断 dual-shift | **已修复**（opportunistic exports + 恢复缺失子包） | 保持 |
| R3 | `extras["apis_coefficient_l2"]` → `intervention_penalty` | **已接线** | 保持 |
| R4 | per-sample valid-intervention mask | **已实现**（采样 + logits 混合 + loss mask） | 保持 |
| R5 | 组件单测 | **通过**（8/8，`pytorch` env） | 回归保留 |
| R6 | 单源单 seed smoke | **通过**（6 epoch 真数据 ADNI→NACC；见 `outputs/journal/dual_shift_apis_v2/smoke/`） | 下一步可开 W3 / 或加长程 E1 |

组件单测命令：

```bat
set PYTHONPATH=%CD%
C:\Anaconda3\envs\pytorch\python.exe -m unittest discover -s tests -p test_apis_v2.py -v
```

## 3. 本地工作包与优先级

```text
W0  文档与分支对齐（本文件）                         ← 完成
W1  接线 R3 + 实现 R4 + 核验 R2                      ← 完成
W2  ADNI→NACC × seed42 × {ce_only, mixstyle, apis_v2} smoke ← 完成（短程 6 epoch）
W3  补 NACC→ADNI seed42 smoke                         ← 完成（短程）；完整 E1 待开
W4  E3 配对机制表（≤30d 主集）构建与 held-out 标记     ← 完成（33/25/73）
W5  负对照（shuffle descriptors）最小子集             ← 下一步可选
```

明确**本机不立即启动**：

- `09` 同机三种子全量（预注册主机为 Linux `an5bi4acenfa1-0`）
- MCI / 三分类 / Joint 补跑 / CDT 性能扩 seed
- 为过 gate 调 APIS 强度、warmup、距离或 CDT 温度

### 3.1 W1 — 编码就绪（进入 GPU 前）

1. 在 `training/dual_shift_loop.py` 将  
   `outputs.extras.get("apis_coefficient_l2")` 传入  
   `compute_dual_shift_loss(..., intervention_penalty=...)`，并写入 train log。
2. 将「无合法 target → 整批 clean」改为 **per-sample mask**：有合法协议的样本走干预路径，其余走 clean；记录 `valid_intervention_frac`。
3. 确认 `experiments/train_journal.py` 能在不依赖缺失 legacy 模块的前提下导入 dual-shift；必要时提供旁路入口。
4. 再跑 `tests/test_apis_v2.py`，并补一条「penalty 进入 total loss」的轻量断言（若已有则扩充）。

### 3.2 W2 — Smoke（单源单 seed）

```bat
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
C:\Anaconda3\envs\pytorch\python.exe run_v2.py --exp journal --direction ADNI_to_NACC ^
  --variants ce_only mixstyle apis_only --seed 42 --device cuda ^
  --output-dir outputs/journal/dual_shift_apis_v2/smoke/seed42/adni_to_nacc ^
  --config_path config/journal_dual_shift_postfix.yaml
```

验收：

- 训练无 NaN；checkpoint 可加载
- log 含 `apis_coefficient_l2` / `valid_intervention_frac`（W1 后）
- 写出 `journal_metrics.json`、`split_manifest.json`、预测 CSV

说明：在 APIS v2 开关落地前，`apis_only` 可能仍指向旧路径；W1 完成后须在配置中显式启用 residual APIS，并在记录中标注代码 commit。

### 3.3 W3 — E1 最小确认包

- 双向 × seeds `42,43,44` × `{ce_only, mixstyle, apis_v2}`
- 输出根：`outputs/journal/dual_shift_apis_v2/e1/`
- 主表：subject-level balanced accuracy + AUC/F1/SEN/SPE；禁止与 legacy Windows/Linux 混表求均值
- metadata / FiLM / unconditional residual 对照在 E1 主包稳定后再排期（对齐 `10` §4）

### 3.4 W4 — 配对机制（E3 准备）

- 主集：ADNI 同受试者 1.5T/3T，最近对 ≤30 天（约 33 人）
- 敏感集：≤7 天（约 25 人）
- 这些 subject **整主体**不得进入训练/选模
- 指标：\|Δp\|、logit 差、翻转率、表征距离，相对 CE/MixStyle

## 4. 分析与结论

### 4.1 结果

| 方法/条件 | 主要指标 | 辅助指标 | 判定 | 备注 |
|---|---:|---:|---|---|
| 组件单测 | 8/8 pass | — | 通过 | W1 |
| R3/R4 接线 | — | — | 通过 | W1 |
| Smoke ADNI→NACC seed42 | AUC/F1 + 审计 | 短程 | 通过 | W2 / `12` |
| Smoke NACC→ADNI seed42 | AUC/F1 + 审计 | 短程；CE 塌缩 | 通路通过 | W3 / `13` |
| 配对 held-out | 33 / 25 / 73 | ≤30d/≤7d/all | 完成 | W4 |
| E1 双向多 seed（完整 epoch） | vs MixStyle & metadata | 校准 | 待跑 | 下一步 |

### 4.2 分析

- 相对基线：legacy APIS 在 Windows 正式 Gate A 已 No-Go；本机重点转向 **APIS v2 主张验证**，不以重复 No-Go 矩阵消耗 GPU。
- 与 seed/方向稳定性：确认性比较至少跨 3 seeds；完整确认性按 `10` 扩至 ≥5。
- 异常与局限：远程分支曾缺 `training/dual_shift_loop.py`；本地已从合并前备份恢复，后续提交须将其纳入版本库以免再丢失。
- 结果结论：待 W1–W2 完成后再写 `review/records/` / `review/analysis/` 文档；本文件只预注册安排。

## 5. 建议下一步实验指导

- 建议动作：**先完成 W1（R3+R4+R2）→ W2 smoke → 再决定是否开 W3 全量 E1**；W4 并行准备配对 held-out。
- 建议依据：`10` §9 明确组件单测不能替代端到端就绪门控；`08` 将环境冲突复现交给 Linux 同机计划 `09`。
- 固定条件：postfix 优化预算与评估聚合方式；subject-mean；source-only checkpoint；禁止 target 调参。
- 进入条件：单测绿；R3/R4 合并进当前分支；`env_fingerprint.json` 已写；输出根为空或显式 force 并记日志。
- 禁止事项：MCI/三分类、Joint 补跑、CDT 多 seed、混用异机 legacy 结果、把 smoke 写成确认性通关。

---

## 附录 A：合并说明

| 项 | 内容 |
|---|---|
| 分支 | `merge/apis-v2-local` |
| 操作 | `git merge origin/feature/apis-v2 --allow-unrelated-histories` |
| 备份 | `_local_pre_merge_backup/`（合并前本地 AdaIN 代码与完整 outputs） |
| 已恢复未跟踪依赖 | `training/`、`scripts/`、`config/`、本地 `outputs/journal/*` |
| 新增关键文档 | `Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`、`review/plans/10_...`、`tests/test_apis_v2.py` |
