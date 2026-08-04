# APIC-V3-FAILURE-DIAG-MCIAD-20260804（5090）

- 依据：`review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`
- 节点：`an5bi4acenfa1-0` · RTX 5090 · torch `2.13.0+cu130`
- Git：`90efdf9`（诊断执行时 HEAD）
- 本地可用产物：**仅 MCI vs AD** primary（CN vs AD checkpoint 在 3090，本机未跑 layer-2）
- 机读归档：`apic_v3_failure_diagnostics_2026-08-04/`

## 0. 执行摘要（判读顺序 §5）

对 MCI 三个代表性 `apic_v3_x` job 的完整 checkpoint 反事实诊断一致指向 **机制类型 1：干预过弱 / 近恒等映射**，并叠加 **基础过拟合**（类型 6），而不是“有害大扰动”（类型 2）。

| 判据 | 观察 | 结论 |
| --- | --- | --- |
| 1. `valid_intervention` 但 layer RMS≈0 | `valid_frac=1`；layer1 RMS ~1e-7–4e-7，layer2 ~8e-5–1.2e-4；JS≈0；flip=0；clean≡shifted 指标 | **成立（主因）** |
| 2. 大 RMS + shifted CE/flip↑ | flip 全为 0；shifted BA=clean BA | **不成立** |
| 3. prototype–label NMI > 扫描属性 | NMI 均很低（≤0.05），label 不高于 field/manufacturer | **不成立** |
| 4. source 扫描 NMI 高、target 仍有害 | 扫描属性 NMI 亦低；且无有害扰动 | **不成立** |
| 5. val composite 与 BA/SEN 不一致 | 无 `val_balanced_accuracy` 日志；选点靠 `val_auc`；seed42 A2N 选到 **epoch 6**（apis_warmup） | **可疑，次要** |
| 6. clean/shifted 均过拟合 | train→~0，val_loss 升到 1.3–1.7；train BA≈0.99 vs val/test 差 | **成立（共病）** |

**本机不做**：seeds 44–46、X+D secondary、按 target 调参。

## 1. Layer-1：训练轨迹汇总

命令（exit 0）：

```bash
python experiments/summarize_apic_v3_diagnostics.py \
  --roots outputs/journal/apic_v3_screening_mci_ad/s1 \
  --output-dir outputs/journal/apic_v3_failure_diagnostics/history
```

产物：`apic_v3_failure_diagnostics_2026-08-04/history/`

### 1.1 Target BA 与相对基线（MCI）

| seed | direction | ce_x | mixstyle_x | apic_v3_x | Δce | Δmix |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | ADNI→NACC | 0.650 | 0.708 | 0.671 | +0.020 | −0.037 |
| 42 | NACC→ADNI | 0.663 | 0.652 | 0.639 | −0.024 | −0.013 |
| 43 | ADNI→NACC | 0.678 | 0.688 | **0.632** | **−0.046** | **−0.056** |
| 43 | NACC→ADNI | 0.629 | 0.648 | 0.634 | +0.005 | −0.014 |

最差单元：**seed43 · ADNI→NACC**（相对两基线均最差）。

### 1.2 APIC 系数与特征强度（apic_v3_x）

| seed | direction | early L2 (ep6–10) | late L2 (last10) | late/early | early feat | late feat | best val_auc ep | final train/val loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | A→N | 7.6e-6 | 6.9e-8 | **0.009** | 6.8e-8 | 3.0e-8 | 6 | 0.004 / 1.75 |
| 42 | N→A | 9.7e-6 | 5.2e-8 | **0.005** | 7.9e-8 | 1.6e-8 | 19 | 0.001 / 1.39 |
| 43 | A→N | 6.5e-6 | 9.6e-8 | **0.015** | 7.4e-8 | 3.2e-8 | 27 | 0.001 / 1.27 |
| 43 | N→A | 1.8e-5 | 5.6e-7 | **0.030** | 2.1e-7 | 6.6e-8 | 12 | 0.002 / 1.39 |

全程 `valid_intervention_frac=1`，但 **L2 与 feature strength 在训练中被压到近零** → 优化把 APIC 退化成近恒等。

### 1.3 Style memory 塌缩（audit）

| seed | direction | valid_slots | max slot share | 备注 |
| --- | --- | ---: | ---: | --- |
| 43 | A→N | 8 | **99.5%** | counts `[…, 8845]` |
| 42 | A→N | 8 | **92.3%** | 选点 epoch 6；诊断 NMI 全 `null`（实质单 prototype 分配） |
| 42 | N→A | 8 | 42.4% | 分配相对更散，但扰动仍≈0 |

## 2. Layer-2：checkpoint 样本级诊断

配置：`journal_dual_shift_apic_v3_screen_mci_ad_remote.yaml`（path remap only；hash 与 manifest 在设置 job seed 后匹配，**未**使用 `--allow-config-hash-mismatch`）。

预检：`seed43/adni_to_nacc --max-samples 32` → exit 0，clean 概率与正式 CSV 差 <1e-5。

完整四 split（exit 0）：

| job | ckpt ep | ckpt SHA-256 (前16) |
| --- | ---: | --- |
| seed43 A→N（最差） | 27 | `c580b54ebb5ddcf6…` |
| seed42 A→N（相对较好，胜 ce） | 6 | `80930580362c680a…` |
| seed42 N→A | 19 | `75ec39b014c3dfec…` |

### 2.1 反事实扰动与 flip（target / source）

三 job × 四 split 共性：

- `prediction_flip_rate = 0`
- `layer1_relative_rms` mean ∈ `[3.7e-8, 3.7e-7]`
- `layer2_relative_rms` mean ∈ `[8.4e-5, 1.2e-4]`
- `js_divergence` mean ~ `1e-9`
- **clean_metrics ≡ shifted_metrics**（BA/AUC/SEN 完全一致）

→ 即使 `alpha_max=0.25` 且 gate≈0.025–0.05，**shifted path 不改变疾病预测**。

### 2.2 Prototype 关联 NMI（全样本）

| job | split | label | field_strength | manufacturer | sequence_family |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed43 A→N | 0.037 | 0.010 | 0.053 | 0.024 | 0.000 |
| seed42 A→N | null | null | null | null | null |
| seed42 N→A | 0.028 | 0.021 | 0.033 | 0.046 | 0.001 |

NMI 低；**不是**“style 混入诊断标签主导”。seed42 A→N 的 null 来自诊断样本几乎落到单一 prototype（分配退化），无法计算双向变异关联。

### 2.3 过拟合信号

以 seed43 A→N 为例：source_train BA **0.985** vs source_val **0.594** / source_test **0.573**；target clean BA 0.625（scan 级汇总；subject-mean 正式表为 0.632）。  
seed42 N→A：train 0.997 vs test 0.653，target sens 仅 0.284。

## 3. 与 CN（3090 记录）的交叉阅读

本机无 CN checkpoint；`review/records/3090` 日志中最差单元 **CN seed43 NACC→ADNI** 同样显示：

- `valid_frac=1.0`
- `apis_l2`：ep6 `7.0e-6` → ep50 `1.3e-6`（衰减，虽不如 MCI 极端）
- 正式表 ΔBA vs ce **−0.174**

与 MCI 机制叙事兼容（干预偏弱 + 任务不稳），但 **CN 的 layer-2 仍须在 3090 原机按 ops/24 用对应 remote YAML 补跑** 后才能并表。

## 4. 综合诊断

1. **主失败模式**：APIC v3 在筛选预算下被优化成 **近恒等残差**（类型 1）。正式推理虽是 clean path，训练期 shift 也几乎不提供可迁移的 style 干预，故相对 MixStyle 无系统性增益并不意外。  
2. **共病**：严重 train/val 分裂（类型 6）使 clean 基线本身不稳；APIC 不能从“更强正则/更强数据增强”路径（MixStyle）中胜出。  
3. **非主因**：有害大扰动（类型 2）、label 泄漏式 style（类型 3）在本机完整诊断中证据不足。  
4. **选点**：`val_auc` composite 可能过早锁定（seed42 A→N ep6）；建议在后续协议修订中评估是否纳入 BA/SEN 约束，但 **当前不得据此改 Gate S1 或调参**。

## 5. 回传清单状态

| 项 | 状态 |
| --- | --- |
| layer1 stdout / exit 0 | ✅ `history/layer1_stdout.txt` |
| `apic_v3_history_summary.csv/json` + epoch csv | ✅ `history/` |
| layer2 stdout / exit 0（3 jobs） | ✅ `layer2_*_stdout.txt` |
| `diagnostic_summary.json` + `sample_diagnostics.csv` | ✅ `checkpoint/*/` |
| config / manifest / ckpt SHA-256 | ✅ `artifact_sha256.json`（ckpt 仅哈希，未入库） |
| GPU / torch / CUDA / git | ✅ `ENVIRONMENT.md` |
| CN layer-2 | ❌ 需 3090 |

## 6. 建议的下一步（不在本轮执行）

- 在 3090 对 CN seed43 N→A / A→N 补跑 layer-2，确认是否同为近恒等。  
- 协议层讨论：提高有效干预下界、memory 均衡约束、或选点指标；**须新 revision**，不得原地改本轮结果。  
- 在修复前不扩 seed、不启 X+D。
