# Dual-Shift APIS 3-seed 执行状态

**创建日期：** 2026-07-29  
**文档序号：** `03`  
**继承：** `01_dual_shift_next_step_decision_2026-07-29.md`、`02_dual_shift_experiment_schedule_2026-07-29.md`  
**状态：** 主队列已完成（2026-07-30 ~02:12）；3-seed 报告已生成  

---

## 1. 本会话已完成

| 项 | 状态 | 路径 / 备注 |
|---|---|---|
| P0 协议冻结 | 完成 | `outputs/journal/dual_shift_postfix/protocol_freeze_2026-07-29/` |
| P2 CDT source 审计 | 完成 | `outputs/journal/dual_shift_apis_3seed/cdt_source_audit_seed42.json` |
| P1 队列接入 | 完成 | `scripts/run_journal_queue.py` stage=`apis_3seed` |
| P1 GPU 队列 | **完成** | 4/4 jobs 均有 `summary.json`；末 job ~07-30 02:12 |
| P1 3-seed 汇总 | 完成 | `gate_report_3seed.json` / `metrics_table_3seed.csv` |
| P1b 配对评估 | 完成 | `outputs/journal/dual_shift_apis_3seed/paired_field_strength_seed42.json` |

---

## 2. CDT 审计结论（seed 42 postfix）

两方向 `cdt_only` **均非 pathological**（`any_pathological=false`）：

| 方向 | n_subjects | ESS ratio | Gini | top1% mass | NaN/Inf |
|---|---:|---:|---:|---:|---|
| ADNI→NACC | 154 | 0.976 | 0.078 | 0.0077 | 无 |
| NACC→ADNI | 576 | 0.626 | 0.308 | 0.023 | 无 |

补充观察：

- checkpoint 中 `cdt.enabled=false`（best 可能落在 warm 期或未启用时刻）；权重本身无极端塌缩。
- ADNI 源权重近均匀；NACC 源中度集中但仍远高于预定义病态阈值（ESS ratio&lt;0.05 或 top1%&gt;0.5）。
- **不触发**“一次稳定化修复”。CDT 仍保持：不做性能扩 seed、不恢复 Joint。

---

## 3. 主队列命令

```bash
python scripts/run_journal_queue.py --stages apis_3seed --device cuda --max-workers 1 --force
```

- config：`config/journal_dual_shift_postfix.yaml`（冻结，不改超参）
- 输出：`outputs/journal/dual_shift_apis_3seed/seed{43,44}/{adni_to_nacc,nacc_to_adni}/`
- 预计墙钟：约 12–18h（串行）

完成后：

```bash
python scripts/report_apis_3seed.py \
  --postfix-root outputs/journal/dual_shift_postfix \
  --seed-root outputs/journal/dual_shift_apis_3seed \
  --seeds 42,43,44 \
  --output-dir outputs/journal/dual_shift_apis_3seed
```

---

## 4. 配对机制验证（seed 42，≤90d，n=21/34 可评估）

| 变体 | mean \|Δp\| | flip rate | Pearson | embed cos |
|---|---:|---:|---:|---:|
| ce_only | 0.00800 | 0 | 0.998 | 0.998 |
| mixstyle | 0.01587 | 0 | 0.991 | 0.995 |
| apis_only | **0.00619** | 0 | 0.999 | 0.996 |

相对差：`apis−ce = −0.0018`，`apis−mixstyle = −0.0097`（负值 = APIS 更一致）。  
三变体均无类别翻转；**未见系统性反证**。注意：有符号 logit 差 APIS 更大（0.80 vs CE 0.43），概率一致性改善不等于 logit 尺度一致。  
覆盖缺口：34 对中仅 21 对在当前 ADNI manifest/图像根下可定位，后续可补路径对齐。

---

## 5. Gate A 预览（尚未判定）

须等 seed 43/44 完成 + 3-seed 汇总 + 配对结果后，按决策 §10.1 / 安排 §3.4 判定。在此之前：

- 不扩 MCI / 三分类  
- 不调 APIS/CDT 超参  
- 不跑 Joint / CDT 多 seed  

---

## 6. 下一步检查点

1. 确认 `apis_3seed` 队列首 job 写入 checkpoint / metrics  
2. 读取配对 JSON，检查 APIS 相对 CE/MixStyle 的 `|Δp|` 是否改善  
3. 队列全部结束后跑 `report_apis_3seed.py` → Gate A Go/No-Go  
