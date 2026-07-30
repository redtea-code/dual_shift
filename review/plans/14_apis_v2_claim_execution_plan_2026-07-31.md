# APIS-V2-CLAIM-EXEC-20260731：APIS v2 主张验证执行计划（校正版）

## 基本信息

- 日期：2026-07-31
- 负责人：本地协作执行
- 状态：计划中（取代 `11` 中与 v2 **主张验证**混用的 legacy Gate A 叙述）
- Git commit：执行前记录；设计基线 `35aa44c`（`feature/apis-v2`）+ 本地 W1 接线
- 关联文档：
  - 规范：`Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`
  - 预注册主张：`review/plans/10_apis_v2_claim_validation_plan_2026-07-30.md`
  - 被校正安排：`review/plans/11_apis_v2_local_experiment_schedule_2026-07-30.md`
  - smoke 记录：`review/records/12_...`、`13_...`（**仅通路证据**）
- 关联数据文件：计划写入 `outputs/journal/dual_shift_apis_v2/claim/`；smoke 产物保留在 `.../smoke/`，不得并入确认性汇总

---

## 0. 校正声明（先读）

### 0.1 本计划回答的唯一问题

在数据约束下，**源域观测协议上的有界有向残差干预（APIS v2）** 是否提升对 **acquisition-protocol / cohort-protocol shift** 的鲁棒性，并在不损害诊断判别的前提下成立。

### 0.2 本计划不回答的问题

- legacy AdaIN APIS 是否通过旧 Gate A（已由 Windows 3-seed 判定 No-Go）
- Linux 同机复现能否改写旧 Gate A（属 `09`，独立于 v2 claim）
- 场强因果效应、MRI 物理仿真、连续 TR/TE/TI 响应面
- MCI / 三分类 / Joint 是否应扩展（当前证据链未到）

### 0.3 已完成 vs 未开始

| 项 | 状态 | 可否计入 claim |
|---|---|---|
| W1 就绪门控（loop / penalty / mask / 导入 / 单测） | 完成 | 进入条件，不是结果 |
| 双向 6-epoch smoke（ce / mixstyle / apis_only） | 完成 | **否**；仅通路 |
| 配对名单 73 / ≤30d 33 / ≤7d 25 | 完成 | 名单可用；**smoke 未排除这些 subject** |
| E1 完整预算外部泛化 | **未开始** | 主判定 |
| E2 协议簇 hold-out | 未开始 | 机制 |
| E3 配对场强评估（训练排除 hold-out） | 未开始 | 机制 |
| 完整基线 + 负对照 + 消融 | 未开始 | claim 成立必要条件 |

**禁止**再把 smoke 的 AUC/F1 表写成“v2 优于 CE/MixStyle”的证据。

---

## 1. 实验指导与依据

- 研究问题：见 §0.1。
- 假设与基线：
  - **主比较**：APIS v2 vs **MixStyle** 与 vs **direct metadata conditioning**。
  - CE、legacy AdaIN APIS、FiLM、unconditional residual、shuffle 等为必要对照/机制对照，**不能替代**上述两项主比较。
- 实施依据：`10` + `APIS_V2_DATA_CONSTRAINED_DESIGN.md`；本地就绪与 smoke 见 `12`/`13`。
- 实验范围：
  - 任务：CN vs AD（与现有 journal binary 映射一致）
  - 队列：ADNI、NACC（NACC 全 3T；反向不能解释为孤立场强机制）
  - 评估层级：E1（性能）→ E3（配对机制，可与 E1 后半并行准备）→ 负对照 → E2/消融
- 判定标准（运行前锁定）：

**E1 主终点（唯一通关标准）**

- 指标：subject-level **balanced accuracy**（先按 subject 聚合扫描，再算指标）
- 成功：在外部测试上，APIS v2 − MixStyle 与 APIS v2 − metadata 的差值均为正，且各自 95% **subject-level 配对 bootstrap CI 不含 0**
- 次要：AUC、SEN、SPE、macro-F1、worst-class recall；如实报告，**不得**用次要指标覆盖主终点失败
- Seeds：确认性至少 **5** 个固定种子（建议 `42,43,44,45,46`）
- Checkpoint / 超参：**禁止**用 target 队列选择；仅 source validation

**停止 / 收缩**

- 主终点双侧（相对 MixStyle 与 metadata）在预注册种子上失败 → 不宣称 v2 claim；可保留机制/负结果文档
- 发现实现错误可修并完整重跑；**禁止** target-driven 调 APIS 强度、warmup、距离、损失权重

---

## 2. 协议冻结（P0，开跑 E1 前必须完成）

### 2.1 配置与输出布局

```text
journal_dual_shift_apis_v2_claim.yaml   # 新建：完整预算 + v2 开关 + 对照清单
outputs/journal/dual_shift_apis_v2/
  smoke/          # 已有；冻结，不覆盖
  paired_holdout/ # 已有名单
  claim/
    env_fingerprint.json
    protocol_freeze/
    e1/
      seed{S}/{adni_to_nacc|nacc_to_adni}/{variant}/
    e3/
    negatives/
    e2/
```

`claim.yaml` 规则：

- 训练预算对齐 postfix（建议 `epochs=50`，`warm_clean/warm_apis` 与 postfix 同量级），**不得**用 6-epoch smoke 预算冒充 E1
- 显式启用 residual APIS v2（非 AdaIN）
- `lambda_intervention` 等写入配置并冻结
- 路径字段可本机 remap；其余超参字节级冻结进 `protocol_freeze/`

### 2.2 配对受试者排除（硬约束）

- 主排除集：`outputs/journal/dual_shift_apis_v2/paired_holdout/paired_holdout_subjects.json` 中 **≤30d** 的 33 名 subject
- 这些 subject 的**全部扫描**不得进入：训练、source validation、checkpoint 选择
- 仅用于 E3 机制评估（及探索性全 73 对分析）
- 进入 E1 前：在 `split_manifest.json` 中断言 hold-out 与 train/val 交集为空

### 2.3 必须实现/接通的代码缺口（E1 前）

| ID | 缺口 | 验收 |
|---|---|---|
| C1 | 训练/划分读取 hold-out subject 列表并排除 | split 断言 + 单测 |
| C2 | 评估主表输出 **balanced_accuracy**（subject_mean） | metrics JSON 含该字段 |
| C3 | `metadata` 条件化基线（与 APIS 同 backbone/预算） | 可经 `--variants metadata` 或等价入口训练 |
| C4 | `film` 基线（已有工厂则接线到 journal dual-shift 公平协议） | 同上 |
| C5 | `legacy_apis`（AdaIN）作为对照变体，**不**与 v2 混名 | 变体名不得再含糊叫 `apis_only` 而不标注版本 |
| C6 | `uncond_residual`（参数量匹配、无协议条件） | 负对照/容量对照 |
| C7 | `apis_v2_shuffle`（描述子打乱） | 机制负对照最小集 |
| C8 | NACC→ADNI 报告按 ADNI 测试子集分 **1.5T / 3T** | 分层 metrics |
| C9 | E1 汇总脚本：逐 seed、跨 seed mean±std、相对 MixStyle/metadata 的 Δ 与配对 bootstrap CI | 单一正式报告根 |

变体命名建议（冻结后不得改）：

```text
ce_only
mixstyle
metadata
film
legacy_apis          # AdaIN
uncond_residual
apis_v2
apis_v2_shuffle      # 负对照（可先最小子集）
# 稍后：cdt_apis_v2 仅当单模块前置满足时
```

### 2.4 环境指纹

启动 E1 前写入 `claim/env_fingerprint.json`：host、Python、torch/CUDA、GPU、git HEAD、config/manifest SHA256、hold-out subject 数与 SHA。

---

## 3. 确认性实验包

### 3.1 E1 — 外部队列泛化（主实验）

**设计**

1. ADNI → NACC：源 ADNI 训练/选模，NACC 一次测试  
2. NACC → ADNI：源 NACC 训练/选模，ADNI 一次测试，并分 1.5T / 3T 报告  

**变体（E1 最小可判决集）**

第一波（必须）：`ce_only`, `mixstyle`, `metadata`, `apis_v2`  

其中 `metadata` 与 `apis_v2` **共享**影像 backbone + 人口学融合（age/sex/education）；  
唯一主差异为协议路径：`metadata` = 采集描述子直接拼接；`apis_v2` = 早期残差协议干预。  
两者均不启用 CDT。`ce_only` 无人口学融合、无协议条件。  
第二波（claim 完整性）：`film`, `legacy_apis`, `uncond_residual`  
第三波（机制最小）：`apis_v2_shuffle`（可先单方向单 seed 再扩）

**种子**：`42,43,44,45,46`  
**聚合**：subject_mean；bootstrap 按 subject  
**产物**：每 run 的 `journal_metrics.json`、预测 CSV、`split_manifest.json`、checkpoint；汇总 `e1/gate_report_claim.json`、`e1/metrics_table_claim.csv`

**E1 Go（宣称性能主结论的最低条件）**

- 两方向均完成 5 seeds，无未解释训练/checkpoint 失败  
- 主终点相对 MixStyle **与** metadata 均满足 §1 的 CI 准则  
- 结论非单 seed 驱动（逐 seed 符号与均值一致）  
- hold-out 排除断言全部通过  

任一方向主终点失败 → **不宣称** v2 相对强对照的稳健优势；进入机制/负结果写作，不扩任务。

### 3.2 E3 — 配对场强机制（支持证据，不替代 E1）

- 集合：≤30d（主）、≤7d（敏感）、全 73（探索，校正时间间隔）  
- 模型：E1 中 **未看见这些 subject** 的 checkpoint（通常取 ADNI 源训练的模型）  
- 指标：跨场强 embedding 距离、\|Δp\|、预测一致/翻转、相对 CE/MixStyle/metadata 的配对改善；检查类间可分性未塌缩  
- 文字口径：复合协议变化下的一致性，**不是**场强因果效应  

### 3.3 负对照（机制）

最小先跑：`apis_v2_shuffle`（batch 内或 subject-wise 打乱，预注册一种并冻结）  
期望：正确描述子优于 shuffle；否则不得把增益归因于协议条件。

### 3.4 E2 与结构消融（E1 通过或明确 No-Go 归档后）

- E2：manufacturer × field × sequence 簇 hold-out；报告 mean / 分散 / worst-cluster；场强分裂称为 **composite protocol shift**  
- 消融：去 `e'-e`、仅 delta、去 RMS cap、单 basis、channel-only、layer1/2、去 coefficient penalty  

---

## 4. 执行优先级（本地）

```text
P0  协议冻结 + C1–C9 缺口闭合 + 指纹          ← 核心缺口已闭合；指纹待开跑前写
P1  E1 第一波：双向 × 5 seeds × 4 变体
    （ce_only, mixstyle, metadata, apis_v2）
P1b 同步：E3 评估脚本对已完成 ADNI 源 ckpt 的配对集
P2  E1 第二波对照：film, legacy_apis, uncond_residual
P3  负对照 apis_v2_shuffle（最小 → 扩 seed）
P4  若 E1 Go：E2 + 结构消融；若 No-Go：写负结果/边界，停止扩任务
```

**明确不启动**

- 用 smoke 数值更新 claim 结论  
- legacy Gate A 同机全量（`09`）与 v2 claim 混表  
- MCI / 三分类 / Joint 性能扩展  
- 查看 target 后改 APIS/CDT 超参  
- 把 NACC→ADNI 改善写成“证明了场强因果机制”

### 4.1 建议启动命令（P1，缺口闭合后）

```bat
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
C:\Anaconda3\envs\pytorch\python.exe experiments\run_apis_v2_claim_e1.py --device cuda
```

若启动器未用，等价展开：

```text
seeds 42..46 × {ADNI_to_NACC, NACC_to_ADNI} × {ce_only, mixstyle, metadata, apis_v2}
--config_path journal_dual_shift_apis_v2_claim.yaml
--output-dir outputs/journal/dual_shift_apis_v2/claim/e1/seed{S}/{direction}
```

汇总：

```bat
C:\Anaconda3\envs\pytorch\python.exe experiments\report_apis_v2_claim_e1.py ^
  --seed-root outputs/journal/dual_shift_apis_v2/claim/e1 ^
  --seeds 42,43,44,45,46 ^
  --output-dir outputs/journal/dual_shift_apis_v2/claim/e1
```

---

## 5. 可复现记录（执行时填写）

- 配置文件：`journal_dual_shift_apis_v2_claim.yaml`
- 数据与划分：ADNI/NACC scan manifest；hold-out JSON；每 run `split_manifest.json`
- 随机种子：42–46
- 环境与硬件：执行前写入 `claim/env_fingerprint.json`
- 启动命令：见 §4.1
- 工作区状态：开跑要求与实验相关路径 clean，或 dirty 列表写入指纹
- 产物位置：`outputs/journal/dual_shift_apis_v2/claim/`

---

## 6. 分析与结果（预留）

### 6.1 结果

| 方法/条件 | balanced_acc（主） | AUC / F1 / SEN / SPE | 判定 | 备注 |
|---|---:|---:|---|---|
| ce_only | 待跑 | 待跑 | 基线 | |
| mixstyle | 待跑 | 待跑 | 主对照 1 | |
| metadata | 待跑 | 待跑 | 主对照 2 | |
| apis_v2 | 待跑 | 待跑 | 方法 | |
| Δ(v2−mixstyle) | 待跑 + CI | — | 主判定之一 | |
| Δ(v2−metadata) | 待跑 + CI | — | 主判定之一 | |

### 6.2 分析

- 相对基线：待 E1 完成后按逐 seed 与跨 seed 报告；**禁止**与 `smoke/` 或 legacy `apis_3seed/` 混算均值  
- 异常与局限：NACC 无 1.5T；配对非同步；场强与厂家/序列纠缠  
- 结果结论：仅当 §1 E1 Go 满足方可写性能主结论；E3/负对照为机制支持

---

## 7. 建议下一步实验指导

- 建议动作：**立即做 P0**（`claim.yaml` + C1–C9），不要先开完整 5-seed GPU 矩阵  
- 建议依据：就绪与 smoke 已满足 `10` §9 的工程进入条件，但 claim 所需终点、对照、hold-out 排除仍未闭合  
- 固定条件：主张边界、主终点、双强对照、5 seeds、source-only 选模、配对排除  
- 进入条件：P0 检查单全部勾选；`protocol_freeze/` 与指纹已写  
- 禁止事项：扩 MCI；混用 legacy Gate A；用 smoke 报 claim；target 调参  

### P0 检查单

- [x] `journal_dual_shift_apis_v2_claim.yaml` 已冻结并归档  
- [x] C1 hold-out 排除进划分 + 断言  
- [x] C2 balanced_accuracy 进正式 metrics  
- [x] C3 metadata 基线可跑（`Model/dual_shift/metadata_baseline.py`，不依赖 Model.comparison）  
- [x] C5 变体命名区分：`apis_only`→`apis_v2`（residual）；`legacy_apis`（AdaIN）未在本分支接线  
- [x] C8 反向 1.5T/3T 分层（`metrics_by_field_strength` + 预测 CSV `field_strength`）  
- [x] C9 汇总与配对 bootstrap 脚本（`experiments/report_apis_v2_claim_e1.py`）  
- [ ] `env_fingerprint.json` 已写（开跑 E1 前填写）  
- [x] 确认 `smoke/` 不参与 claim 汇总（报告脚本拒绝 smoke 路径）  

**P0 代码状态（2026-07-31）**：仅在 `feature/apis-v2` 之上叠加 claim 缺口与 journal dual-shift 运行所需最小 `training/` 辅助模块；不纳入 Model.comparison / dictionary / gamma 配置或旧脚本套件。

---

## 附录 A：与旧文档关系

| 文档 | 角色 |
|---|---|
| `10` | 主张与实验设计规范（上位） |
| `11` | 早期本地安排；其中 Gate A / 短程数值叙事 **由本文校正** |
| `09` | legacy 同机复现；与 v2 claim **并行独立**，禁止混表 |
| `12`/`13` | smoke 通路记录；非确认性 |

## 附录 B：工作量粗估（单卡 3090 量级）

- E1 第一波：2 方向 × 5 seeds × 4 变体 × ~1–1.5 h ≈ **40–60 h**  
- 第二波 3 变体：再约 **30–45 h**  
- E3 / 负对照：以评估与少量重训为主，远小于 E1  

串行执行；有空卡可按 seed 或方向并行，但须保证配置与指纹一致。
