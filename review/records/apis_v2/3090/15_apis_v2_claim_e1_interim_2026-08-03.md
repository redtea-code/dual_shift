# APIS v2 Claim E1 中期实验记录（供审阅）

**日期：** 2026-08-03  
**状态：** 中期 / **非正式 claim 冻结**（矩阵未完成）  
**代码：** `f8df40b`（`fix: harden APIS v2 claim experiment isolation`）  
**运行分支：** `run/apis-v2-claim-e1-f8df40b`  
**协议修订：** `claim.protocol_revision = 2`  
**配置：** `journal_dual_shift_apis_v2_claim.yaml`  
**输出根：** `outputs/journal/dual_shift_apis_v2/claim/e1/`  
**关联计划：** `review/plans/14_apis_v2_claim_execution_plan_2026-07-31.md`、`review/plans/10_apis_v2_claim_validation_plan_2026-07-30.md`

---

## 0. 审阅结论（先读）

| 问题 | 中期判断 |
|---|---|
| 可否宣布 E1 claim 通过？ | **否**。预注册 5 seeds × 双向未齐；正式 report gate 未跑。 |
| 相对 MixStyle 是否有正向趋势？ | **有限正向、不稳定**。ADNI→NACC（4 seeds）mean ΔBA ≈ +0.030（赢 3/4）；NACC→ADNI（3 seeds）mean ΔBA ≈ +0.051（赢 2/3）。 |
| 相对 metadata / metadata_xda？ | **不可解释为机制胜利**。两类 metadata 在已完成 job 上普遍塌缩到 BA≈0.50。 |
| 协议隔离（hold-out / split_seed）？ | **中期检查通过**（见 §3）。 |
| 下一步 | 等队列跑完 seed45/nacc 余下变体 + seed46 双向后，再跑 `report_apis_v2_claim_e1.py` 做正式门控。 |

---

## 1. 实验目的与预注册口径

### 1.1 唯一问题

在数据约束下，**源域观测协议上的有界有向残差干预（APIS v2）** 是否相对主基线提升外部队列上的 **subject-level balanced accuracy**。

### 1.2 主比较（协议 r2）

| 变体 | 输入轴 | 角色 |
|---|---|---|
| `ce_only` | X | 纯 CNN 对照 |
| `mixstyle` | X+D | **主基线 1** |
| `metadata` | X+A | 补充：直接采集元数据条件化 |
| `metadata_xda` | X+D+A | **主基线 2（公平轴）** |
| `apis_v2` | X+D（训练期 A 残差） | 主张方法 |

主终点：target 上 subject-mean **balanced_accuracy**。  
成功标准（预注册）：相对 MixStyle **与** metadata_xda 的 Δ 均为正，且各自 95% subject-level 配对 bootstrap CI 不含 0（需齐全 5 seeds）。

### 1.3 协议冻结要点（r2）

1. Hold-out（`subjects_le_30d`，33 人）在 **6/2/2 之前**从 source 池剔除。  
2. E1 target 与 E3 paired 强制不相交；manifest 按 cohort 分别记录。  
3. `split_seed=42` 冻结划分；`training_seed∈{42…46}` 仅控制初始化/采样。  
4. 各变体重建同 seed 的 subject-balanced sampler；metadata* 与 dual-shift **共用**选模规则。  
5. 产物要求 `claim_protocol_revision=2`；旧 r1 / smoke **不计 claim**。

---

## 2. 运行与完成度（截稿时）

**环境指纹（队列写入）：** host 本机 · Python/torch 见 `claim/env_fingerprint.json` · GPU RTX 3090 · git `f8df40b` · `split_seed=42` · revision `2`。

| Job | Variants | 状态 |
|---|---|---|
| seed42 ADNI→NACC | 5/5 | 完成 |
| seed42 NACC→ADNI | 5/5 | 完成 |
| seed43 ADNI→NACC | 5/5 | 完成 |
| seed43 NACC→ADNI | 5/5 | 完成 |
| seed44 ADNI→NACC | 5/5 | 完成 |
| seed44 NACC→ADNI | 5/5 | 完成（中断后整 job 重开） |
| seed45 ADNI→NACC | 5/5 | 完成 |
| seed45 NACC→ADNI | 3/5 | **进行中**（截稿时 `metadata_xda` ~ep44） |
| seed46 双向 | 0/5 | 排队中 |

合计（metrics 落盘）：**38 / 50** variant；完整双向 seed：**3**（42–44），ADNI→NACC 另有 seed45。

**队列备注：** 双进程并行两次在 NACC→ADNI `ce_only` ~ep32 异常退出；当前以 `max_workers=1` 续队列，整 job 重开中断任务（不续跑 checkpoint）。

---

## 3. 协议健康检查

| 检查项 | 结果 |
|---|---|
| `split_seed` 跨 training seed 一致 | ✅ ADNI→NACC seeds 42–45 训练集 subject 集合相同 |
| Hold-out ∩ NACC→ADNI E1 target | ✅ 0（target n=229；E3 ADNI paired=28） |
| ADNI→NACC E1 target | n=960；E3：ADNI 28 / NACC 0 |
| `claim_protocol_revision` | ✅ 已完成 metrics 均为 2 |
| smoke 混入 | ✅ 输出在 `claim/e1/`，与 `smoke/` 隔离 |

---

## 4. 主表：target subject-level balanced accuracy

### 4.1 ADNI→NACC（完整 4 seeds：42–45）

| Seed | ce_only | mixstyle | metadata | metadata_xda | apis_v2 | Δ Mix | Δ XDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.820 | 0.752 | 0.500 | 0.500 | **0.823** | +0.071 | +0.323 |
| 43 | 0.833 | 0.797 | 0.500 | 0.500 | 0.771 | −0.026 | +0.271 |
| 44 | 0.593 | 0.793 | 0.500 | 0.500 | **0.816** | +0.022 | +0.316 |
| 45 | 0.734 | 0.745 | 0.500 | 0.500 | **0.798** | +0.053 | +0.298 |
| **mean±std** | 0.745±0.111 | 0.772±0.027 | 0.500±0 | 0.500±0 | **0.802±0.023** | **+0.030** | +0.302 |

- 对 MixStyle：赢 **3/4**（唯一失败 seed43）。  
- CE 方差大（seed44 明显塌向高 SPE / 低 SEN）。  
- metadata / metadata_xda **全部 BA=0.50**（塌缩）。

### 4.2 NACC→ADNI（完整 3 seeds：42–44）

| Seed | ce_only | mixstyle | metadata | metadata_xda | apis_v2 | Δ Mix | Δ XDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.772 | 0.806 | 0.473 | 0.513 | 0.805 | −0.0005 | +0.292 |
| 43 | 0.802 | 0.789 | 0.511 | 0.500 | **0.862** | +0.073 | +0.362 |
| 44 | 0.821 | 0.725 | 0.500 | 0.501 | **0.806** | +0.081 | +0.305 |
| **mean±std** | 0.798±0.025 | 0.773±0.043 | 0.494±0.020 | 0.505±0.007 | **0.824±0.033** | **+0.051** | +0.320 |

- 对 MixStyle：赢 **2/3**（seed42 基本打平）。  
- metadata* 仍接近随机 / 单类预测，**不能**支撑“战胜公平元数据基线”的叙述。

### 4.3 次要指标（完整子集，apis_v2）

| 方向 | Seed | AUC | macro-F1 | SEN | SPE |
|---|---:|---:|---:|---:|---:|
| A→N | 42 | 0.904 | 0.769 | 0.812 | 0.835 |
| A→N | 43 | 0.877 | 0.775 | 0.624 | 0.919 |
| A→N | 44 | 0.885 | 0.787 | 0.753 | 0.879 |
| A→N | 45 | 0.886 | 0.749 | 0.769 | 0.828 |
| N→A | 42 | 0.907 | 0.795 | 0.721 | 0.890 |
| N→A | 43 | 0.930 | 0.855 | 0.814 | 0.910 |
| N→A | 44 | 0.899 | 0.799 | 0.752 | 0.860 |

（正式论文表应在 5 seeds 齐全后由 report 脚本统一重算 + bootstrap CI。）

---

## 5. 现象与风险（需审阅关注）

### 5.1 Metadata 塌缩（P0 级解读风险）

- ADNI→NACC：已完成 4 seeds 上 `metadata` 与 `metadata_xda` 的 target BA **恒为 0.50**（全预测 CN 或全预测 AD）。  
- NACC→ADNI：BA 亦在 ~0.47–0.51。  
- 因此 **Δ(apis − metadata_xda) 虽大，但不能计入 claim 机制解释**，只说明“直接 A / X+D+A concat 基线当前失败”。  
- 建议后续单独开诊断：学习率/融合尺度、acquisition encoder fit、collapse_guard 是否把选模推向退化解等。

### 5.2 相对 MixStyle：方向对、力度不稳

- 跨方向均值略正，但存在明确负例（A→N seed43；N→A seed42 平局）。  
- 在 5 seeds 与配对 bootstrap 完成前，**不得**写成“显著优于 MixStyle”。

### 5.3 运行稳定性

- 双 worker 并行两个 NACC→ADNI 时，两次在 ~epoch 32 无 metrics 落盘即退出。  
- 单 worker 重开后 `seed44/nacc` 已完整跑通，说明更像资源/会话问题而非单 job 必现逻辑错误；仍建议正式跑维持 `max_workers=1` 或加强进程监护。

### 5.4 尚未覆盖

- seed45/nacc 剩余变体、seed46 双向  
- 正式 `report_apis_v2_claim_e1.py`（分层 bootstrap CI、split/config 一致性门控）  
- E2 / E3 机制评估  
- metadata 修复后的重跑（若认定当前 metadata* 无效，需决定是否重训该变体轴）

---

## 6. 建议的审阅决议项

请审阅后勾选/批注：

1. **中期是否接受“相对 MixStyle 有弱正向趋势”的内部叙述？**（是 / 否 / 需等 5 seeds）  
2. **metadata / metadata_xda 塌缩：是否阻塞 claim 叙事，必须先修再重训？**（阻塞 / 不阻塞但降级为失效基线）  
3. **未完成 job 继续单进程跑完即可，还是 seed45/46 全部强制重开？**（续当前队列 / 全量重开）  
4. **正式门控前是否要求 metadata_xda 达到非塌缩最低门槛（例如 target BA 显著 > 0.55）？**

---

## 7. 复现入口（供核对）

```text
# 配置与启动
journal_dual_shift_apis_v2_claim.yaml
experiments/run_apis_v2_claim_e1.py   # protocol_revision==2, max_workers<=2

# 正式汇总（矩阵齐全后）
experiments/report_apis_v2_claim_e1.py --seeds 42,43,44,45,46

# 产物
outputs/journal/dual_shift_apis_v2/claim/e1/seed{S}/{adni_to_nacc|nacc_to_adni}/{variant}/
  journal_metrics.json
  target_predictions.csv
  ../split_manifest.json
outputs/journal/dual_shift_apis_v2/claim/env_fingerprint.json
```

---

## 8. 变更相对 r1 的说明

相对早期污染跑次（hold-out 进入 N→A target、五套不同划分、metadata 仅 X+A 却与 APIS 不公平比较），本中期表 **仅基于 r2 产物**：

- Hold-out 已从 E1 target 剔除（N→A n=229）  
- 固定 `split_seed=42`  
- 增加 `metadata_xda`；选模规则对齐  
- 旧 smoke / r1 数字 **未并入**本表  

---

**文档用途：** 内部审阅与决策，**不是**最终 claim 报告。齐种子后应以 report 脚本输出替换 §4 汇总，并补 CI。
