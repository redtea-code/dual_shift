# APIS-V2-CLAIM-MCIAD-E1-INTERIM-20260803：MCI vs AD E1 中期判断（5090 节点）

## 基本信息

- 日期：2026-08-03
- 负责人：远程 Linux `an5bi4acenfa1-0`（6× RTX 5090）
- 状态：运行中（中期快照；非正式 Gate）
- Git commit：执行分支 `feature/apis-v2-claim-mci-ad`（启动器含 3-GPU 槽位队列）
- 关联文档：`review/plans/16_apis_v2_claim_e1_mci_ad_remote_plan_2026-08-02.md`、`review/plans/14_apis_v2_claim_execution_plan_2026-07-31.md`
- 关联数据文件：`outputs/journal/dual_shift_apis_v2/claim_mci_ad/e1/`
- 配置：`journal_dual_shift_apis_v2_claim_mci_ad_remote.yaml`（`protocol_revision=2`，label `2→MCI/0`，`3→AD/1`）

---

## 1. 队列进度（截至 2026-08-03 上午）

| 状态 | Job |
|---|---|
| 已完成（5/5 变体 + summary） | seed42 ADNI→NACC；seed42 NACC→ADNI；seed43 ADNI→NACC |
| 进行中（3 卡） | seed43 NACC→ADNI；seed44 ADNI→NACC；seed44 NACC→ADNI |
| 排队 | seed45、seed46 双向 |

- 并行：`max_workers=3`，`CUDA_VISIBLE_DEVICES` 槽位 `0,1,2`；单 job 结束后自动接下一个
- 单 job 墙钟约 **10–11 h**（5 变体串行）
- 预计全部完成约还需 **1.5–2 天**（粗估）

日志：`outputs/journal/queue_logs/claim_mci_ad_e1_20260802_082501.log`

---

## 2. 已完成 target 指标（subject-level）

主终点：**balanced_accuracy**。次要：AUC / macro-F1 / SEN / SPE。

### 2.1 ADNI→NACC

| seed | ce_only | mixstyle | metadata | metadata_xda | apis_v2 |
|---|---:|---:|---:|---:|---:|
| 42 bAcc | 0.6353 | 0.6363 | 0.5000* | 0.5000* | 0.6045 |
| 42 AUC | 0.6911 | 0.7055 | 0.5000 | 0.5000 | 0.7621 |
| 42 F1 | 0.6191 | 0.6394 | 0.4005 | 0.4005 | 0.6023 |
| 43 bAcc | 0.6770 | 0.6606 | 0.5000* | 0.5000* | 0.6016 |
| 43 AUC | 0.7398 | 0.7397 | 0.5000 | 0.5000 | 0.7524 |
| 43 F1 | 0.6287 | 0.6642 | 0.4005 | 0.2493 | 0.5977 |

### 2.2 NACC→ADNI

| seed | ce_only | mixstyle | metadata | metadata_xda | apis_v2 |
|---|---:|---:|---:|---:|---:|
| 42 bAcc | 0.6434 | 0.6415 | 0.5232 | 0.5188 | 0.5561 |
| 42 AUC | 0.6965 | 0.7329 | 0.4931 | 0.5209 | 0.7017 |
| 42 F1 | 0.6381 | 0.6445 | 0.3905 | 0.4642 | 0.5426 |

\* `metadata` / `metadata_xda` 在 ADNI→NACC 上出现类塌缩（SEN≈0 或 SPE≈0），bAcc≈0.5。

### 2.3 APIS v2 相对主对照（已完成 seed×dir）

| seed / 方向 | ΔbAcc (v2−mixstyle) | ΔbAcc (v2−metadata_xda) | ΔAUC (v2−mixstyle) |
|---|---:|---:|---:|
| 42 ADNI→NACC | −0.0318 | +0.1045 | +0.0566 |
| 42 NACC→ADNI | −0.0854 | +0.0373 | −0.0312 |
| 43 ADNI→NACC | −0.0590 | +0.1016 | +0.0127 |

---

## 3. 中期判断（非正式，不可作 claim）

1. **相对 MixStyle（主对照 1）**：已完成 3 个 seed×direction 上，APIS v2 的 **balanced_accuracy 均更低**；AUC 在 ADNI→NACC 上偶有升高，但主终点暂不利。
2. **相对 metadata_xda（主对照 2）**：bAcc 差值多为正，但 ADNI→NACC 上对照明显塌缩，**不能单独支撑**“优于公平 X+D+A 基线”的主张。
3. **metadata 轴异常**：需在完整矩阵结束后单独审计（训练崩溃 vs 协议/标签问题），不得用塌缩基线美化 Δ。
4. **正式结论条件未满足**：需 seeds **42–46** 双向齐全后，运行 `experiments/report_apis_v2_claim_e1.py`，按预注册 CI 准则判定 Go/No-Go。
5. **禁止事项（维持）**：不与他机 CN vs AD `claim/` 混表；不用本中期表宣称通关；不看 target 调参；不扩 MCI vs CN / 三分类。

---

## 4. 下一步

- 保持 3-GPU 队列跑完剩余 jobs
- 全齐后生成 `claim_mci_ad/e1/gate_report_claim.json` 与 metrics 表
- 另写正式 analysis 文档（本文件仅中期 records）
