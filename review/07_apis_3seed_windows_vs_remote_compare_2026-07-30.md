# APIS 3-seed：本机 Windows vs 远程 Linux 结果对比

**日期：** 2026-07-30  
**文档序号：** `07`  
**合并分支：** `results/apis_3seed_remote_43_44` → `main`  

---

## 1. 证据分轨（不可混用）

| 来源 | 路径 | Seeds | 说明 |
|---|---|---|---|
| Windows 本机队列 | `outputs/apis_3seed/`（汇总）+ 训练机 `dictionary_learn/outputs/journal/dual_shift_apis_3seed/` | 42+43+44 | Gate A 初判所用；含 seed42 postfix |
| Linux 远程 | `outputs/journal/dual_shift_apis_3seed/` | **仅 43/44** | 分支 `results/apis_3seed_remote_43_44`；无 seed42；无 checkpoint |

配置名义均为 postfix 冻结协议；远程有路径 remap（`journal_dual_shift_postfix_remote.yaml`）与少量工程适配（见 `05_dual_shift_remote_claim_2026-07-29.md`）。**数值不可当作同一次训练的重复。**

---

## 2. APIS vs CE 关键指标对照（apis_only）

| seed / 方向 | Win AUC | Remote AUC | Win F1 | Remote F1 |
|---|---:|---:|---:|---:|
| 43 ADNI→NACC | 0.8891 | 0.8999 | 0.7604 | 0.7805 |
| 43 NACC→ADNI | 0.8947 | **0.9287** | **0.6973** | **0.8462** |
| 44 ADNI→NACC | 0.8977 | 0.8976 | 0.7793 | 0.7876 |
| 44 NACC→ADNI | 0.9465 | 0.9423 | 0.7496 | 0.8325 |

最大分歧在 **seed43 NACC→ADNI**：Windows APIS F1 明显劣于 CE（gate fail）；远程 APIS AUC/F1 均优于 CE（该方向 gate pass）。

---

## 3. Gate 口径对照

| 口径 | Windows（`outputs/apis_3seed/gate_report_3seed.json`） | Remote（仅 43/44） |
|---|---|---|
| ADNI→NACC × {43,44} | pass / pass | pass / pass |
| NACC→ADNI × {43,44} | **43 fail** / 44 pass | **pass / pass** |
| 含 seed42 的双边 Gate A | **No-Go**（42 正向 F1 fail + 43 反向 fail） | 未含 42；预览写「43/44 全过」≠ 正式 Gate A |

**合并后正式 Gate A 仍以「seed42 postfix + 可复现的 43/44」完整包为准；远程 43/44 作为独立复现/敏感性证据保留，不得覆盖 Windows 汇总表而不加标注。**

---

## 4. 合并动作

- 保留 `main` 上已有 Gate A 报告与 `outputs/apis_3seed/`、`outputs/postfix/`。
- 并入远程完整 metrics/predictions/summaries 与认领文档。
- 不删除任一侧结果；后续若选定「官方 43/44」主机，需预注册并重跑 seed42 同机汇总。
