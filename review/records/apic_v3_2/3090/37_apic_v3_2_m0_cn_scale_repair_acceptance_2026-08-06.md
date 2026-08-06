# APIC-V3_2-M0-CN-ACCEPT-20260806：3090 尺度修复验收（CN）

## 基本信息

- 日期：2026-08-06
- 节点：Windows · RTX 3090
- 工作树：`D:\cyh\apic_v3_2_m0_fix`
- 分支 / commit：`fix/apic-v3-2-m0` @ `cee93138cc95b12690d6a7db32bd079d73f142bc`
- 本地补丁（未提交）：`Model/dual_shift/apis_v3_2.py` 中 `finalize_style_bank` 的 CPU/CUDA 设备对齐（否则 B0→apis_warmup 无法 finalize）
- 验收依据：`review/analysis/36_apic_v3_2_m0_scale_repair_acceptance_2026-08-05.md`
- 关联：plan 26 Gate M0；ops 24 诊断流程；既有 r4 CN checkpoints（`dictionary_learn/outputs/journal/apic_v3_2_screening_cn_ad/r4/`）
- `formal_run_allowed`：**仍为 false**（本记录不构成 E3，不授权扩 seed / X+D）

## 1. 验收要求（doc 36）摘要

1. B0：每个 K=4 slot 同时具备 fit 与 calibration subject 支持，缺失立即失败。
2. 每个有效 source slot 至少有一个相对距离落在冻结 band `[delta_min, delta_max]=[0.50, 3.00]` 的替代 prototype。
3. E2 单元：supported fraction 与 nonzero realized RMS fraction 均非零；报告 layer1/layer2 RMS band hit rate。
4. 不支持样本 shifted logits/embedding 严格回退 clean；shifted BN 不更新 running stats。
5. checkpoint reload 的 clean 概率与正式导出预测在 `1e-5` 内一致。
6. layer-2 诊断覆盖一个成功与一个失败单元，记录 clean/shift BA、flip、JS、slot counts、assignment-label NMI。
7. 本地：`tests/test_apis_v3_2.py`。

## 2. 实际执行命令

### 2.1 单元测试

```text
C:\Anaconda3\envs\pytorch\python.exe -m pytest tests/test_apis_v3_2.py -q --tb=short
# exit 0；9 passed
```

### 2.2 Layer-2 checkpoint 诊断（smoke → full，四 CN 单元）

配置：`journal_dual_shift_apic_v3_2_screen_cn_ad.yaml`  
checkpoint 根：`d:\cyh\dictionary_learn\outputs\journal\apic_v3_2_screening_cn_ad\r4\`  
（`--allow-config-hash-mismatch`：仅绝对路径/工作树差异）

| 单元 | smoke `--max-samples 32` | full |
|---|---|---|
| seed42 ADNI→NACC | EXIT=0 | EXIT=0 |
| seed42 NACC→ADNI | EXIT=0 | EXIT=0 |
| seed43 ADNI→NACC | EXIT=0 | EXIT=0 |
| seed43 NACC→ADNI | EXIT=0 | EXIT=0 |

示例：

```text
python experiments/export_apic_v3_checkpoint_diagnostics.py \
  --config journal_dual_shift_apic_v3_2_screen_cn_ad.yaml \
  --job-dir <r4>/seed42/adni_to_nacc \
  --variant apic_v3_2_x --device cuda --batch-size 2 \
  --allow-config-hash-mismatch \
  --output-dir outputs/journal/apic_v3_2_m0_acceptance/checkpoint/seed42_adni_to_nacc_full
```

### 2.3 B0 bank-build smoke（真实 ADNI source）

临时配置：`journal_dual_shift_apic_v3_2_m0_b0_smoke_cn.yaml`（epochs=7, warm_clean=5, warm_apis=1）

```text
python experiments/train_journal.py \
  --config_path journal_dual_shift_apic_v3_2_m0_b0_smoke_cn.yaml \
  --direction ADNI_to_NACC --seed 42 --variants apic_v3_2_x \
  --device cuda \
  --output-dir outputs/journal/apic_v3_2_m0_b0_smoke/cn_adni_to_nacc \
  --force-variants
# EXIT_B0=0
```

短 warm-clean（1 epoch）时 finalize 曾因簇/校准支持不足失败；对齐协议 warm_clean=5 后通过。  
未修复设备对齐前，apis_warmup 入口 `finalize_style_bank` 会因 CPU/CUDA 混算直接崩溃。

## 3. 结果对照 doc 36

### 3.1 本地单测

| 项 | 结果 |
|---|---|
| `tests/test_apis_v3_2.py` | **PASS**（9/9），含相对尺度不变性、真实替代 target、空 calibration 严格失败、零支持损失、revision-4 权重 |

### 3.2 条件 2：相对距离 band 内替代 prototype

对四个 r4 CN `apic_v3_2_x` checkpoint，用 `style_prototypes` / `prototype_radii` 计算相对距离（半径均值归一化）：

| 单元 | valid slots | 每 slot 是否存在 band 内替代 |
|---|---|---|
| seed42 A→N | 4/4 | **是** |
| seed42 N→A | 4/4 | **是** |
| seed43 A→N | 4/4 | **是**（slot1 仅 1 个替代，仍非空） |
| seed43 N→A | 4/4 | **是** |

### 3.3 条件 3–6：四单元 layer-2 诊断（源侧为主）

机器可读汇总：`apic_v3_2_m0_cn_acceptance_2026-08-06/support_rms_summary.csv`

**所有四单元 × 四 split：**

- supported fraction **> 0**
- supported 样本 layer1/layer2 nonzero RMS fraction = **1.0**
- supported 样本 layer1 band `[0.001,0.03]` / layer2 band `[0.003,0.05]` hit rate = **1.0**
- unsupported：layer RMS nonzero = **0**；clean vs shifted 概率差 max = **0**；embedding cosine ≤ **~2e-7**
- reload clean 概率 vs 正式预测：全部 split `matches_within_1e_5=true`（max abs diff ≤ 1.2e-7）
- shifted path：诊断脚本 `counterfactual_only` + 模型 `update_bn_stats=False` / paired clean BN moments（代码路径，与 ops 24 一致）

**成功 / 失败对照单元（条件 6）：**

| 角色 | 单元 | source_val clean/shift BA | flip | JS mean | slots | label NMI |
|---|---|---|---|---|---|---|
| 相对更强（A→N seed42） | seed42_adni_to_nacc | 0.809 / 0.809 | 0 | ~3.0e-5 | [18,14,28,33] | 0.114 |
| 相对更弱（N→A seed43） | seed43_nacc_to_adni | 0.785 / 0.785 | 0 | （见 summary） | [122,34,70,103] | 0.080 |

源侧 support fraction（source_val）：0.925 / 0.819 / 0.774 / 0.855。  
注：plan 26 Gate M0 条目 2 要求 calibration/source-val support ∈ `[0.20,0.90]`；seed42 A→N source_val=**0.925** 略超上界。doc 36 条件 3 仅要求非零——本轮按 doc 36 记 **PASS**，但完整 Gate M0 仍需单独裁定。

### 3.4 条件 1：B0 fit+calibration

- 单测 `test_v3_2_strict_bank_rejects_missing_calibration`：**PASS**
- 真实数据 B0 smoke：**PASS**（EXIT=0）
  - epoch 7 `apis_warmup`：`style_memory_valid_slots=4`，`valid_intervention_frac≈0.898`，`prototype_relative_separation≈1.376`（落在 `[0.50, 3.00]`）
  - 证明 K=4 在严格 fit+calibration 门控下可完成 finalize，且干预非零
  - 说明：performance selector 仍可能把 `best_checkpoint` 定在 bank 之前的 clean epoch；机制证据以 epoch 历史 / live bank 为准，不以该 early checkpoint 的空 `style_valid` 否定 B0

## 4. 产物路径

- 诊断输出：`outputs/journal/apic_v3_2_m0_acceptance/checkpoint/`
- 归档：`review/records/apic_v3_2/3090/apic_v3_2_m0_cn_acceptance_2026-08-06/`
  - `support_rms_summary.csv`
  - `*_diagnostic_summary.json`
  - `checkpoint/<unit>/{diagnostic_summary.json,sample_diagnostics.csv}`
- B0 smoke：`outputs/journal/apic_v3_2_m0_b0_smoke/`（及 stdout）
- 临时 B0 配置：`journal_dual_shift_apic_v3_2_m0_b0_smoke_cn.yaml`（本地验收用，非正式 protocol）

## 5. 判定

| Gate / 文档 | 判定 |
|---|---|
| doc 36 尺度修复验收（CN / 3090 可覆盖范围） | **PASS**（条件 1–6 + 单测） |
| plan 26 完整 Gate M0（含 MCI 四源单元 + support∈[0.20,0.90]） | **未完全关闭** |
| 正式 E3 / `formal_run_allowed=true` | **仍禁止** |

## 6. 建议下一步

1. 提交（或另开 PR）`finalize_style_bank` 设备对齐修复，并入 `fix/apic-v3-2-m0`。
2. 5090 上对 MCI×{ADNI,NACC} 复跑同等 layer-2 / B0 验收，凑齐 E2b 四源单元。
3. 审视 seed42 A→N source_val support 0.925 是否触发 plan 26 M0 上界；必要时在正式 E1/E2 冻结流程中处理，**不要**为追 target 调参。
4. CN+MCI 机制门关闭后，再考虑解除 `formal_run_allowed` 并启动 revision-4 E3（CE/MixStyle 同协议重跑）。
