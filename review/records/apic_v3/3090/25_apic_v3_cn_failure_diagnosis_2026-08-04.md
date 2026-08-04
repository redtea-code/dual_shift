# APIC-V3-CN-FAILDIAG-20260804：3090 CN_vs_AD 失败机制诊断

## 基本信息

- 日期：2026-08-04
- 节点：本机 Windows RTX 3090
- 依据：`review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`
- Git commit（诊断时）：`90efdf9`（含诊断脚本）；本机额外修补 `summarize_apic_v3_diagnostics.py` 的 Python 3.8 `removeprefix` 兼容
- 配置：`journal_dual_shift_apic_v3_screen_cn_ad.yaml`
- 范围：仅本机已完成的 **CN_vs_AD** primary（MCI 产物不在本机）
- 关联主表：`review/records/apic_v3/3090/23_apic_v3_cn_ad_s1_metrics_2026-08-04.md`
- 产物目录：`review/records/apic_v3/3090/apic_v3_failure_diagnostics_2026-08-04/`（四 job 完整 layer-2；旧 `apic_v3_failure_diagnostics/` 仅 seed43 过渡副本）

## 1. 环境与指纹

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| PyTorch / CUDA | 1.13.0+cu117 / 11.7 |
| Python | 3.8.19 |
| config SHA-256 | `81d05ece9128f6fb92f00126fcfcbbe5634c022faaccf4961e5d8e7ab92abbe0` |
| config_hash（脚本校验） | `2148a6b5b5a5d46a99ec0f5589271e1fe6b100fb7c4fbb51fab0851bc03eba3f` |
| seed43 N→A checkpoint SHA-256 | `b6596284056965f39f215f6fb7d1a9b1a27e2b9f62534a4b13682b9b7edb0d6c` |
| seed43 A→N checkpoint SHA-256 | `8613fb45ae8838b6…`（完整见对应 `diagnostic_summary.json`） |
| seed43 N→A manifest SHA-256 | `070eda361cacb9d7835182f7e3181e565bfc5ccf2dc44e6a6026a286feba7ea6` |
| seed43 A→N manifest SHA-256 | `e245a9b3206c928b…` |

Checkpoint 本体未入库（约 100MB×2）；SHA-256 已记录。

## 2. 执行记录（ops 24）

| 步骤 | 命令要点 | 退出码 | stdout |
|---|---|---:|---|
| L1 历史汇总 | `summarize_apic_v3_diagnostics.py --roots .../cn_ad/s1` | 0 | `stdout/summarize_stdout.txt` |
| L2 预检 | `export_... --job-dir .../seed43/nacc_to_adni --max-samples 32` | 0 | `stdout/precheck_seed43_nacc_stdout.txt` |
| L2 全量 | seed42/43 × 双向，四 split | 0 | `apic_v3_failure_diagnostics_2026-08-04/layer2_*_stdout.txt` |

参考预测一致性：两 job 的 source_val / source_test / target 重建 clean 概率均 `matches_within_1e_5=true`。

## 3. 第一层：训练轨迹（CN 12-run）

来源：`history/apic_v3_history_summary.csv`。

### 3.1 apic_v3_x 关键量

| seed | direction | target BA | early L2 (ep6–10) | late L2 (last10) | late/early | late feat strength | max composite epoch |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | adni_to_nacc | 0.7908 | 8.57e-6 | 5.28e-8 | 0.006 | 1.35e-8 | 12 |
| 42 | nacc_to_adni | 0.7302 | 1.22e-5 | 2.84e-6 | 0.232 | 2.07e-7 | 26 |
| 43 | adni_to_nacc | 0.7862 | 8.24e-6 | 6.32e-8 | 0.008 | 1.63e-8 | 31 |
| 43 | nacc_to_adni | 0.6233 | 1.56e-5 | 2.72e-5 | 1.742 | 1.62e-6 | 22 |

判读：

- 三个相对较好的单元上，**晚期 APIC L2 相对早期大幅衰减**（late/early ≪ 1），干预系数趋向消失。
- 最差单元 seed43 N→A 的 late/early > 1，但 **绝对 feature strength 仍仅 ~1e-6**，不足以构成有效残差。
- 终局 `train_loss≈1e-3`、source 近完美 vs target 明显掉点，符合 **普通过拟合 / 队列外推失败** 与 **APIC 近恒等** 并存。

## 4. 第二层：checkpoint 反事实（ops 24；CN 四 job 齐套）

四个 `seed×direction` 均已跑 **完整四 split**（预检后去掉 `--max-samples`）。归档：
`apic_v3_failure_diagnostics_2026-08-04/checkpoint/seed{42|43}_{adni_to_nacc|nacc_to_adni}/`。

### 4.0 四 job target 对照

| job | ckpt ep | target clean BA | shifted BA | flip | gate | L1 RMS | L2 RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed42_adni_to_nacc | 12 | 0.7881 | 0.7881 | 0.0000 | 0.0271 | 5.13e-08 | 7.40e-05 |
| seed42_nacc_to_adni | 26 | 0.7353 | 0.7353 | 0.0000 | 0.0266 | 2.98e-07 | 9.16e-05 |
| seed43_adni_to_nacc | 31 | 0.7837 | 0.7837 | 0.0000 | 0.0306 | 3.15e-08 | 6.23e-05 |
| seed43_nacc_to_adni | 22 | 0.6270 | 0.6289 | 0.0018 | 0.0396 | 1.18e-06 | 1.16e-04 |

共性：gate≈0.03、layer RMS / JS≈0、clean≡shifted（最差单元仅 flip≈0.002）→ **ops 24 条款 1（近恒等）在全部 CN job 成立**。

### 4.1 最严重失败：CN seed43 NACC→ADNI（ckpt epoch 22）

| split | n | clean BA | shifted BA | flip rate | gate mean | layer1 RMS | layer2 RMS | emb cos dist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source_train | 716 | 0.9548 | 0.9548 | 0 | 0.0402 | 1.12e-6 | 1.15e-4 | ~1e-8 |
| source_val | 227 | 0.8171 | 0.8171 | 0 | 0.0403 | 1.13e-6 | 1.15e-4 | ~1e-8 |
| source_test | 238 | 0.8201 | 0.8201 | 0 | 0.0402 | 1.12e-6 | 1.15e-4 | ~1e-8 |
| target | 547 | 0.6270 | 0.6289 | 0.0018 | 0.0396 | 1.18e-6 | 1.16e-4 | ~1e-8 |

附加：

- `style_delta` mean ≈ 0.16，`style_entropy` ≈ 0.97，但 **condition_gate≈0.04** 把有效扰动压到接近 0。
- 训练 memory counts：`[8,2,4,3,180,4,11970,1]` → **slot6 垄断赋值**。
- 诊断样本硬分配：`prototype_slot` **1728/1728 全为 slot4** → `prototype_assignment_nmi` 全为 `null`（左侧取值 <2，按脚本定义不可算）。

对应 ops 24 §5：**条款 1 成立**——干预日志非空，但 layer 相对 RMS / embedding 距离 / JS ≈ 0，APIC 退化为近恒等；**不是条款 2（有害扰动）**。

### 4.2 唯一双胜基线：CN seed43 ADNI→NACC（ckpt epoch 31）

| split | n | clean BA | shifted BA | flip rate | gate mean | layer1 RMS | layer2 RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_train | 334 | 1.000 | 1.000 | 0 | 0.0311 | 1.7e-8 | 4.5e-5 |
| source_val | 106 | 0.818 | 0.818 | 0 | 0.0316 | 2.0e-8 | 4.8e-5 |
| source_test | 107 | 0.861 | 0.861 | 0 | 0.0313 | 1.5e-8 | 4.2e-5 |
| target | 1181 | 0.784 | 0.784 | 0 | 0.0306 | 3.2e-8 | 6.2e-5 |

memory counts：`[4,4,1,1,4,4,5,8609]` → 同样 **单 slot 垄断**。  
即使该单元相对 CE/MixStyle 胜出，**反事实路径仍几乎不改变预测**——收益更可能来自常规训练波动/选模，而非可检验的 APIC 残差机制。

## 5. 按 ops 24 判读顺序的结论

| # | 判据 | 本机 CN 结论 |
|---|---|---|
| 1 | valid 干预但 layer RMS≈0 | **成立（主因）**：gate 压制 + 残差近恒等 |
| 2 | RMS 大且 shifted 变差 | **不成立**：shifted≈clean，flip≈0 |
| 3 | prototype–label NMI > 扫描 NMI | **无法计算**：硬分配塌缩为单 slot |
| 4 | source 扫描 NMI 高但 target 有害 | **不适用**：扰动本身近零 |
| 5 | val composite 与 BA/SEN 不一致 | **部分相关**：N→A 选 ep22 时 val SEN≈0.70，target SEN≈0.39 |
| 6 | clean/shifted 均过拟合 | **成立**：train/source 高、target（尤其 N→A）崩 |

**综合归类（失败主因）：干预过弱 / 近恒等退化 + 基础跨队列过拟合；不是“强而有害的 APIC 扰动”。**

因此：在完成机制修复前，**不增加 seeds 44–46，不启动 X+D secondary，不按 target 调参**（与 ops 24 §6 一致）。

## 6. 建议下一步（诊断导向，非调参）

1. 检查并约束 style-memory 塌缩（单 slot 垄断）与过低 `condition_gate`。
2. 在合成/小规模设定验证：当 gate/RMS 有下界时，shifted path 是否能产生可测 flip/JS。
3. 重新评估 checkpoint 目标是否过度偏好 source-val composite，而牺牲 target SEN。
4. MCI 节点应用同一 ops 24 流程后，再决定是否合并 Gate S1。

## 7. 回传文件清单

- [x] `apic_v3_failure_diagnostics_2026-08-04/history/*`
- [x] `checkpoint/seed42_{adni_to_nacc,nacc_to_adni}/{diagnostic_summary.json,sample_diagnostics.csv}`
- [x] `checkpoint/seed43_{adni_to_nacc,nacc_to_adni}/{diagnostic_summary.json,sample_diagnostics.csv}`
- [x] `layer2_*_stdout.txt` / smoke stdout（退出码均为 0）
- [x] `artifact_sha256.json`、`ENVIRONMENT.md`
- [x] config / manifest / checkpoint SHA-256
- [ ] MCI roots：见 `review/records/apic_v3/5090/`（本机不跑）
