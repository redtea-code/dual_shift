# Dual-Shift 远程认领与启动记录（本机）

**日期：** 2026-07-29  
**节点：** Linux `/zjs/AD_Project/dual_shift` · env `cyh` · GPU×6  
**依据：** `review/operations/04_dual_shift_remote_handoff_2026-07-29.md`（主）、`review/records/03_dual_shift_apis_3seed_status_2026-07-29.md`、`review/plans/01_dual_shift_next_step_decision_2026-07-29.md`

---

## 1. 交接结论（执行口径）

| 项 | 决定 |
|----|------|
| 主任务 | **T1 APIS 3-seed**：seed **43/44** × 双向 × `{ce_only, mixstyle, apis_only}` |
| 配置 | 冻结 postfix；本机仅路径 remap → `config/journal_dual_shift_postfix_remote.yaml` |
| 禁止 | MCI / Joint / CDT·dual_shift 的 43/44 / 看 target 调参 / 混 Stage2 |
| 正式证据 | postfix seed42（需同步）+ 本机训出的 `dual_shift_apis_3seed/` |
| Gate A | 四 job 齐后 `report_apis_3seed.py` |

分片方案：**A（本机整包）** — 无其他远程节点争用本输出目录。

---

## 2. T0 环境验收

| 检查 | 结果 |
|------|------|
| `cyh` + CUDA | ✅ torch 2.13+cu130，6×5090 |
| 图像数据 | ✅ `./dataset/{ADNI,NACC}_dataset` → Denosising |
| 临床 CSV | ✅ `./dataset/{ADNI,NACC}.csv` |
| 远程 config（仅改路径） | ✅ `journal_dual_shift_postfix_remote.yaml`（`scan_manifest.root: ./scan_manifests`） |
| 队列 | ✅ `apis_3seed`；Python=`cyh`；`max-workers`≤**3**；按 GPU 隔离 |
| **`scan_manifests/{ADNI,NACC}_scan_manifest.csv`** | ✅ 已上传；`image_path` 已 remap 到本机绝对路径（原 Windows `F:\...` 备份为 `*.winpaths.bak`） |
| 可用样本 | ✅ ADNI n=791（CN/AD）；NACC n=1181 |
| `outputs/journal/dual_shift_postfix/`（seed42） | ❌ 未随仓库复制（汇总 Gate A 时需要；不阻塞开训 43/44） |

---

## 3. 路径/代码调整（为跑通实验，非超参改动）

1. `config/journal_dual_shift_postfix_remote.yaml`：`scan_manifest.root` → `./scan_manifests`；`data/scan_manifests` 软链同目录。
2. `data/journal_dataset.py`：本地 glob 已找到文件时，不再因 Windows `image_path` 缺失而误排除。
3. `Model/comparison/{__init__,factories}.py`：`torch_geometric` 相关模型改为可选/懒加载（journal 路径不需要）。
4. `scripts/run_journal_queue.py`：smoke / apis_3seed 使用 remote config；默认 Python=`cyh`；`MAX_WORKERS_CAP=3`。

---

## 4. 任务认领表（本节点）— **已点火**

| Job ID | seed | direction | GPU | 状态 |
|--------|------|-----------|-----|------|
| J43A | 43 | ADNI_to_NACC | 0 | 训练中（2026-07-29 06:31） |
| J43N | 43 | NACC_to_ADNI | 2 | 训练中 |
| J44A | 44 | ADNI_to_NACC | 1 | 训练中 |
| J44N | 44 | NACC_to_ADNI | — | 排队（max-workers=3） |

- 启动命令日志：`outputs/journal/queue_logs/apis_3seed_20260729_063155.log`
- PID：见同目录 `.pid`
- 输出根：`outputs/journal/dual_shift_apis_3seed/`

预计：4 job × 3 变体；3 卡并行约 **~8–12h**。

完成后：

```bash
$JOURNAL_PYTHON scripts/report_apis_3seed.py
```

（Gate A 完整汇总仍需 postfix seed42 产物。）

---

## 5. 已做准备

- [x] 阅读 `04` / `03` 交接  
- [x] dataset 软链  
- [x] remote config（路径-only）  
- [x] 队列配置  
- [x] scan_manifests 上传 + image_path remap  
- [x] smoke_dual_shift  
- [x] T1 主训启动  
- [ ] T4 Gate A 汇总（等四 job + seed42）

---

## 6. 一句话

**scan_manifests 已就绪并完成 Linux 路径 remap；`apis_3seed`（seed 43/44 × 双向）已在 3 张 GPU 上启动训练。**
