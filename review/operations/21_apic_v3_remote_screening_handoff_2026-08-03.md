# APIC-V3-S1 远程筛选交接

## 基本信息

- 日期：2026-08-03
- 负责人：cyh / 远程实验协作者
- 状态：代码准备完成；真实数据 smoke 待远程环境执行
- Git commit：提交后填写
- 关联文档：`review/plans/20_apic_v3_image_only_comparison_plan_2026-08-03.md`
- 关联数据文件：运行后写入 `outputs/journal/apic_v3_screening_*/`

## 1. 运行前检查

1. 工作区必须 clean，并记录当前 Git commit。
2. 检查两个 YAML 中 `F:/ADNI/...` 与 `F:/NACC/...` 是否对应远程机器真实只读数据；只能做路径映射，不得修改模型、split、标签、Gate 或训练超参数。
3. 确认 `data/claim/paired_holdout_subjects.json` 含 `subjects_all_paired` 73 人集合。
4. 每张 GPU 同时最多一个 worker。
5. `X` 主分析通过前不得启动 `X+D` 或 seeds 44--46。

## 2. 配置预检

```bash
python experiments/run_apic_v3_screening.py --phase primary --fingerprint-only
```

该命令冻结两个任务的配置和环境指纹，不启动训练。若失败，先修复配置身份或远程路径，不得改用 APIS v2 claim launcher。

## 3. Smoke

先在 CN vs AD、ADNI->NACC、seed42 上运行三个 `X` 变体的 one-epoch smoke。远程执行记录必须包含 checkpoint reload、target clean inference、memory valid slots、`valid_intervention_frac`、NaN 和单类预测检查。

正式矩阵不能使用 smoke 输出，也不能使用 smoke epoch 预算。

## 4. 两 seed 主矩阵

单 GPU：

```bash
python experiments/run_apic_v3_screening.py \
  --phase primary \
  --gpu-ids 0 \
  --max-workers 1
```

两张 GPU：

```bash
python experiments/run_apic_v3_screening.py \
  --phase primary \
  --gpu-ids 0,1 \
  --max-workers 2
```

launcher 固定执行两个任务、两个方向、seeds 42/43 和 `{ce_x,mixstyle_x,apic_v3_x}`。已有结果只有在 config hash、模态和协议身份完全匹配时才会跳过；不完整 job 会完整重跑该 job 的三个变体。

## 5. Gate S1

全部主矩阵完成后运行：

```bash
python experiments/report_apic_v3_screening.py
```

结构化输出：

```text
outputs/journal/apic_v3_screening_gate_s1.json
outputs/journal/apic_v3_screening_gate_s1.csv
```

仅当 JSON 中 `pass=true` 时，才允许运行：

```bash
python experiments/run_apic_v3_screening.py \
  --phase secondary \
  --gpu-ids 0,1 \
  --max-workers 2
```

## 6. 回传清单

- 两个 `protocol_freeze/` 目录；
- 每个 job 的 `split_manifest.json`、`metadata_match_audit.json` 和日志；
- 六个变体的 `journal_metrics.json`、逐 subject predictions 和 checkpoint；
- `paired_comparisons.json`、`summary.json`；
- Gate S1 JSON/CSV；
- 失败 job 的退出码、完整日志和 GPU/驱动状态。

## 7. 禁止事项

- 不根据 target 指标选择 epoch、阈值或超参数；
- 不把 `X+D` 与 `X` 跨模态比较解释为 APIC 收益；
- 不用两 seed 宣称统计显著；
- 不删除失败、OOM、NaN 或单类预测结果；
- 不覆盖 APIS v2 r2 的配置和输出目录。
