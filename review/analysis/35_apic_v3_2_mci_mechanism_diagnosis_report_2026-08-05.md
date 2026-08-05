# APIC v3_2 MCI 机制诊断报告：不宜进入正式 E3

## 结论

对 **成功单元**（seed42 NACC→ADNI，唯一双胜）与 **失败单元**（seed42 ADNI→NACC，最差 −mix）做完整 layer-1/2 诊断后：两侧均为 **全程零干预**（gate/RMS/JS/flip=0，`valid_intervention=0`）。根因是 `delta_max=0.5` 与 PCA 原型间距（~18–52）尺度不匹配，替代原型选择恒失败。  
**建议：不进入正式 E3**；先修机制尺度并完成 Gate M0。与 CN 3090 诊断同构。

## 1. 选取与执行

| 角色 | 单元 | 依据 |
|---|---|---|
| 成功 | seed42 · NACC→ADNI | 唯一同时胜过 CE 与 MixStyle |
| 失败 | seed42 · ADNI→NACC | 最大负 ΔBA vs MixStyle |

- Layer-1：`summarize_apic_v3_diagnostics.py` on MCI r4（exit 0）
- Layer-2：`export_apic_v3_checkpoint_diagnostics.py --variant apic_v3_2_x` 全四 split（两单元均 exit 0，各 1322 samples）
- 归档：`review/records/apic_v3_2/5090/apic_v3_2_mci_failure_diagnostics_2026-08-05/`
- 详表：`review/records/apic_v3_2/5090/33_apic_v3_2_mci_mechanism_diagnosis_2026-08-05.md`

## 2. 关键发现

1. **Bank 表面健康**：4/4 valid slots，counts 分散，不再是 v3 式单槽垄断。
2. **干预实际关闭**：训练日志与 checkpoint 反事实一致——gate/RMS/JS/flip 全 0。
3. **尺度 bug**：`delta_min/max` 按 ~0.5 量级裁剪，而原型欧氏距离为十几到几十；choices 为空 → target=src → 全部 invalid。
4. **性能不可归因于 APIC**：所谓成功/失败差异是在无效干预下的 clean-path 波动。

## 3. 与 Gate / E3

| 问题 | 答案 |
|---|---|
| 是否近恒等？ | 是（支持掩码全关导致的恒等） |
| 是否有害大扰动？ | 否（无扰动） |
| Gate M0？ | 不通过（support / RMS） |
| 正式 E3？ | **否** |

## 4. 下一步

1. 重新定义并冻结相对位移 / delta band（与 PCA 度量一致），写入新机制配置。
2. 用 E1 合成与 E2 source-only 证明 supported 比例与 RMS band。
3. 通过 E2b 四单元 Gate M0 后，再考虑正式 revision-4 E3（CE/MixStyle 需同协议重跑）。

在此之前，保持 `formal_run_allowed: false`，不扩 seed、不启 X+D。
