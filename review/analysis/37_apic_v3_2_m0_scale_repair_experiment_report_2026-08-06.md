# APIC v3_2 M0 尺度修复实验报告（5090 · MCI · 2026-08-06）

## 结论

本轮在 5090 上完成 **MCI · ADNI↔NACC · seed42** 的 M0 尺度修复验收。机制尺度在 A2N（成功单元）上 **可工作**；N2A 在 post-bank `last_checkpoint`（epoch7 / epoch50）上 **BA 塌缩持续**（~0.5）。  
**裁决：NO-GO。** `formal_run_allowed: false`；非正式 E3；不扩 seed、不启 X+D。

机读归档：`review/records/apic_v3_2/5090/apic_v3_2_m0_acceptance_2026-08-06/`（`GO_NO_GO.md` / `m0_metrics_summary.json`）。

## 1. 动机与修复边界

### 动机

前序诊断（analysis 35 / records 33）确认：bank 表面健康（K=4 有计数），但 `delta_max≈0.5` 与 PCA 原型欧氏间距（十几到几十）尺度不匹配，替代原型恒失败 → **全程零干预**。性能差异不能归因于 APIC 机制。

### 修复边界（对照 analysis 36）

已合入 / 本轮本地补齐的机制与工程修复仅覆盖：

| 项 | 说明 |
|---|---|
| 相对 delta band | target 合法性按两端 calibration radius 归一化（`cee9313`） |
| calibration floor | 小校准集上为每有效 slot 保底分配，不搬运 fit/k-means 成员 |
| radii / device align | bank 构建时 CPU 观测张量与 CUDA buffer 对齐 |
| style_bank DataLoader | bank 构建用全覆盖 loader，避免 WeightedRandomSampler 饿死 slot |
| `last_checkpoint.pt` | 每 epoch 落盘，便于 selector 锁 clean-best 时审计 post-bank 权重 |
| conditional teacher reload | best 可能来自 clean warmup（无 teacher keys）或机制阶段 |
| 单测 | calibration floor 等；`tests/test_apis_v3_2.py` **10 passed** |

**不在边界内**：正式 E3、扩 seed、X+D、解除 `formal_run_allowed`。

## 2. 实验设置

| 项 | 值 |
|---|---|
| 机器 | 5090；repo `/zjs/AD_Project/dual_shift_github` |
| 分支 | `fix/apic-v3-2-m0` |
| 诊断基线 git | `cee93138cc95b12690d6a7db32bd079d73f142bc` |
| 配置 | `journal_dual_shift_apic_v3_2_m0_mci_ad_remote.yaml`（归档 `configs/`） |
| 任务 | MCI vs AD；方向 ADNI→NACC 与 NACC→ADNI；**仅 seed42**（CN 本轮不做） |
| 标准 | `review/analysis/36_apic_v3_2_m0_scale_repair_acceptance_2026-08-05.md` |
| 环境 | conda `cyh`；torch/cuda 见 layer-2 `diagnostic_summary.json` |

Layer-2 覆盖：A2N `best_checkpoint` epoch **47**；N2A post-bank snapshot epoch **7** 与 epoch **50**。大文件 `.pt` 不入库，见 `ARTIFACT_PATHS.json`。

## 3. 单测

```text
conda run -n cyh python -m pytest tests/test_apis_v3_2.py -q
..........                                                               [100%]
10 passed in 2.64s
```

日志：归档 `unit_tests/pytest_apis_v3_2.txt`。

## 4. E2b 发布性能（selector / journal）

数值均来自归档 `metrics/*_summary.json`（subject-mean BA）。

| 单元 | 发布 best epoch | target BA | audit valid_slots | 备注 |
|---|---:|---:|---:|---|
| ADNI→NACC | **47** | **0.6265** | **4** | 含 bank+teacher；机制健康单元 |
| NACC→ADNI clean-best（历史） | **3** | **0.7036** | **0** | 无 bank；非机制主张 |
| NACC→ADNI retrain 发布 best | **4** | **0.6366** | **0** | 仍为 clean warmup；机制审计用 last_checkpoint |

要点：**N2A selector 持续偏好 clean-warmup best**；发布 BA 不能代表 post-bank 机制状态。

## 5. Layer-1 / Layer-2 与 doc 36 scorecard

### 5.1 M0 条件对照

| # | Condition（doc 36） | A2N | N2A ep7 last | N2A ep50 last |
|---|---|---|---|---|
| 1 | K=4 slot fit+calibration | **PASS** 4/4；counts `[28,8,26,33]` | **PASS** 4/4；`[80,56,71,47]` | **PASS** 同左 |
| 2 | 每有效 source slot 有 in-band 替代 | **PASS**；supported 中 band hit=1.0 | **PASS** | **PASS** |
| 3 | supported / nonzero RMS 非零 | **PASS** supported∈[~0.41,0.79] | **PASS** ∈[~0.49,0.93] | **PASS** |
| 4 | unsupported → clean fallback | **PASS** | **PASS** | **PASS** |
| 5 | reload clean 与导出 ≤1e-5 | **PASS**（max abs ~1e-7） | **N/A**（snapshot 旁无参考 CSV） | **N/A** |
| 6 | Layer-2 覆盖成功+失败 | **PASS（pair）** | BA 塌缩 | 塌缩仍在 |

### 5.2 A2N layer-2（n=1322，checkpoint epoch 47）

来源：`layer2/a2n/diagnostic_summary.json`。

| split | clean BA | shift BA | flip | JS mean |
|---|---:|---:|---:|---:|
| source_train | 0.8822 | 0.8822 | 0.0000 | 3.00e-05 |
| source_val | 0.5904 | 0.5827 | 0.0092 | 2.34e-05 |
| source_test | 0.4957 | 0.4957 | 0.0000 | 5.25e-05 |
| target | 0.6175 | 0.6178 | 0.0046 | 1.55e-05 |

Assignment NMI：split **0.054**，label **0.022**，field_strength **0.062**，manufacturer **0.040**。

判读：相对修复前「全程零干预」，A2N 已出现非零 gate/RMS/JS 与微小 flip；target clean≈shift BA，机制扰动可控。

### 5.3 N2A clean-best vs last_checkpoint

| 视图 | epoch | target BA（或 layer-2） | bank | 机制可读性 |
|---|---:|---:|---|---|
| 历史 published clean-best | 3 | **0.7036**（journal） | valid_slots=0 | 仅 clean 路径 |
| retrain published best | 4 | **0.6366**（journal） | valid_slots=0 | 仍非机制 |
| last_checkpoint epoch7 | 7 | layer-2 clean=shift **0.5000**；flip=0；JS=0 | 4/4 | supported 预测全 class 0 |
| last_checkpoint epoch50 | 50 | layer-2 target clean **0.5052** / shift **0.5014**；flip **0.0045**；JS **5.01e-05** | 4/4 | BA 仍塌缩；预测多为 class 1 |

epoch7 关键（`layer2/n2a_epoch7/`）：source supported ≈0.91–0.93，target ≈0.49；band hit among supported =1.0；**nonzero RMS ≠ 健康决策**（JS/flip/BA 全塌）。  
epoch50（`layer2/n2a_epoch50/`）：bank 仍 4/4；BA 仍约 **0.48–0.52**；JS 变为极小非零；**失败模式未解除**。

## 6. 结论与下一步

1. **尺度修复部分成功**：A2N 证明相对 band + calibration floor 可使 K=4 合法、supported/RMS 非零，且 1e-5 预测一致性通过。
2. **Gate M0 仍不通过**：doc 36 要求完整 E1/E2 清零后才能解除正式阻断；本轮仅为 seed42 双向证据，且 N2A 失败单元 BA 塌缩持续到 epoch50。
3. **政策不变**：`formal_run_allowed=false`；无 E3；不扩 seed、不启 X+D。
4. **建议下一步**  
   - 针对 N2A：区分 selector（clean-best）与机制审计（`last_checkpoint`）；调查 post-bank 决策塌缩（全 class0→多 class1）。  
   - 闭环 N2A snapshot 的 1e-5 参考预测路径。  
   - 在失败模式可解释/可缓解前，勿把发布 BA 登记为机制收益。

## 7. 关联

- 验收标准：`review/analysis/36_apic_v3_2_m0_scale_repair_acceptance_2026-08-05.md`
- 前序诊断：`review/analysis/35_apic_v3_2_mci_mechanism_diagnosis_report_2026-08-05.md`
- 本轮归档：`review/records/apic_v3_2/5090/apic_v3_2_m0_acceptance_2026-08-06/`
- CN 并行记录（3090）：`review/records/apic_v3_2/3090/37_apic_v3_2_m0_cn_scale_repair_acceptance_2026-08-06.md`（编号同属 37 族，目录不同）
