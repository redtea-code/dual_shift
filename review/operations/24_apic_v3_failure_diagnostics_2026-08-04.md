# APIC v3 失败机制诊断脚本使用说明

## 1. 目的和边界

本诊断回答以下问题：

1. APIC 的残差干预是否在训练后趋近于零；
2. style memory 的 prototype 是否主要关联扫描属性，还是关联诊断/队列；
3. counterfactual shift 是否改变疾病预测或破坏类别敏感度；
4. 失败来自干预过弱、干预有害、checkpoint 选择，还是普通过拟合。

诊断不会更新 style memory，不会改变正式结果，也不得使用 target 诊断结果调节本轮
Gate S1。checkpoint 脚本在 `eval()` 和 `no_grad()` 下构造反事实 shifted path；该路径只用于
机制审计，正式 target 指标仍使用 clean inference。

## 2. 需要回传的产物

每个 `seed x direction` job 至少保留：

- 三个变体的 `journal_metrics.json`；
- `source_val_predictions.csv`、`source_test_predictions.csv`、`target_predictions.csv`；
- job 根目录的 `split_manifest.json` 和 `metadata_match_audit.json`；
- `apic_v3_x/best_checkpoint.pt`；
- 实际训练使用的路径映射 YAML 和 `protocol_freeze/`。

checkpoint 通常较大，不要求提交到 Git；可放共享存储，但必须同时记录文件 SHA-256。

## 3. 第一层：汇总现有训练轨迹

该脚本不加载图像或checkpoint，可以先在结果汇总机器运行：

```bash
python experiments/summarize_apic_v3_diagnostics.py \
  --roots \
    outputs/journal/apic_v3_screening_cn_ad/s1 \
    outputs/journal/apic_v3_screening_mci_ad/s1 \
  --output-dir outputs/journal/apic_v3_failure_diagnostics/history
```

输出：

- `apic_v3_history_summary.csv/json`：每个变体的最佳validation composite、最终损失、
  APIC早晚期L2比值和target指标；
- `apic_v3_epoch_history.csv`：完整epoch轨迹，包括style confidence、entropy、delta、gate、
  feature strength和memory assignments。

旧run没有loss分解字段时相应列为空。当前代码之后产生的新run还会记录 `clean_ce`、
`shift_ce`、`js`、`feature_consistency` 和 `intervention_penalty`。

## 4. 第二层：checkpoint样本级诊断

必须使用该job训练时的**准确路径映射YAML**。脚本默认校验config hash，不允许误用另一台机器
或另一个seed的配置。

先在最严重失败单元上对每个选中split最多做32个样本的预检：

```bash
python experiments/export_apic_v3_checkpoint_diagnostics.py \
  --config journal_dual_shift_apic_v3_screen_cn_ad.yaml \
  --job-dir outputs/journal/apic_v3_screening_cn_ad/s1/seed43/nacc_to_adni \
  --device cuda \
  --batch-size 2 \
  --max-samples 32
```

预检通过后移除 `--max-samples`，执行完整四个split。建议至少诊断：

```text
CN_vs_AD seed43 NACC_to_ADNI  # 最严重失败
CN_vs_AD seed43 ADNI_to_NACC  # 唯一同时胜过两个基线
```

也可以只选择部分split：

```bash
python experiments/export_apic_v3_checkpoint_diagnostics.py \
  --config /exact/path/to/journal_dual_shift_apic_v3_screen_mci_ad_remote.yaml \
  --job-dir outputs/journal/apic_v3_screening_mci_ad/s1/seed42/adni_to_nacc \
  --splits source_train source_val target \
  --device cuda
```

输出位于 `JOB_DIR/apic_v3_x/diagnostics/`：

- `sample_diagnostics.csv`：逐样本style向量、prototype分配概率、gate、真实layer1/layer2
  相对RMS扰动、clean/shifted概率、label flip、CE和JS；
- `diagnostic_summary.json`：各split的clean/shifted指标、扰动分位数、flip rate，以及prototype
  assignment与split、label、field strength、manufacturer、sequence family的NMI；同时检查
  重建的clean概率是否在 `1e-5` 内匹配正式prediction文件。

`--allow-config-hash-mismatch` 仅用于已经人工核对“差异只有绝对路径”的情况。正式诊断优先保存
并使用原始remote YAML，不应常规绕过hash检查。

## 5. 判读顺序

1. `valid_intervention=1` 但layer相对RMS接近0：优化把APIC退化成近恒等映射；
2. layer RMS明显、shifted CE/flip显著升高：干预破坏诊断信息；
3. prototype-label NMI高于扫描属性NMI：style表征混入疾病/解剖内容；
4. source扫描属性NMI高、target扰动仍有害：source style support不能覆盖target；
5. validation composite最佳epoch与BA/敏感度明显不一致：checkpoint选择目标需要重新评估；
6. clean和shifted均严重过拟合：先处理基础训练稳定性，不应只调整APIC alpha。

NMI只用于描述关联，不是因果证据。单一场强或单一类别的稀疏分层不计算机制结论。

## 6. 回传清单

- 两个诊断脚本的stdout和退出码；
- `apic_v3_history_summary.csv/json`、`apic_v3_epoch_history.csv`；
- 代表性成功/失败单元的 `sample_diagnostics.csv` 和 `diagnostic_summary.json`；
- config、manifest和checkpoint SHA-256；
- GPU、PyTorch、CUDA及Git commit。

完成上述诊断前，不增加seeds 44--46，不启动X+D secondary，不根据target结果调参。
