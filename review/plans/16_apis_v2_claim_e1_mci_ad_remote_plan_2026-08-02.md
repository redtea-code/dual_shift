# APIS-V2-CLAIM-E1-MCIAD-REMOTE-20260802：远程 MCI vs AD 确认性 E1

## 基本信息

- 日期：2026-08-02
- 负责人：远程 Linux `an5bi4acenfa1-0`
- 状态：计划中（取代本机原定 CN vs AD E1；CN vs AD 已在其他机器执行）
- Git commit：基于 `feature/apis-v2-claim-p0`；本计划随 launcher 多任务输出根支持一并提交
- 关联文档：`review/plans/14_...`、`15_...`（CN vs AD）；`Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`；`01` 中 Gate B「单一 MCI 先导」口径
- 关联数据文件：`outputs/journal/dual_shift_apis_v2/claim_mci_ad/`（**禁止**写入或混用 `.../claim/`）

---

## 0. 任务重分配声明

| 项 | 决定 |
|---|---|
| 本机原计划（`15`） | CN vs AD E1 → **取消本机执行**，避免与他机重复 |
| 本机新任务 | **MCI vs AD** 双向外部泛化（APIS v2 claim 协议） |
| 与他机关系 | 他机继续 CN vs AD；两边输出树与报告 **完全隔离**，不得拼表求均值 |
| 与 legacy Gate A | 独立；不因旧 AdaIN No-Go 改超参；本任务是 **v2 claim 的任务轴扩展**，不是 legacy Gate B 复活 |

---

## 1. 实验指导与依据

- 研究问题：在 **MCI vs AD** 上，APIS v2（源域观测协议有界残差干预）是否相对 MixStyle 与 metadata_xda 提升 subject-level balanced accuracy（跨队列）。
- 假设与基线：同 CN vs AD claim —— 主比较 `apis_v2` vs `mixstyle` **与** vs `metadata_xda`；CI 不含 0。
- 实施依据：算力避免重复；`01` 允许在方法冻结后选 **一个** MCI 先导；本机预注册选定 **MCI vs AD**（非 MCI vs CN）。
- 实验范围：
  - 标签：folder `2→0 (MCI)`, `3→1 (AD)`；CN 不进入训练集
  - 方向：ADNI→NACC、NACC→ADNI
  - Seeds：训练 `42–46`；`split_seed=42`
  - 变体：`ce_only`, `mixstyle`, `metadata`, `metadata_xda`, `apis_v2`
  - 预算：50 epoch；`alpha_max=0.25`；`protocol_revision=2`
  - Hold-out：仍排除 ≤30d 配对 33 subjects（为后续 E3；与任务轴无关的硬约束）
- 判定标准：
  - 主终点：subject-level **balanced_accuracy**
  - Go：两方向上 Δ(v2−mixstyle) 与 Δ(v2−metadata_xda) 均为正且 95% 配对 bootstrap CI 不含 0
  - 失败 → 不宣称 MCI 轴 claim；不改开 MCI vs CN「碰运气」；不扩三分类
  - 禁止 target 调参

### 1.1 样本量（manifest `pre_label`，开跑前再核 `split_manifest`）

| 队列 | MCI 扫描 | AD 扫描 | 合计扫描 | 约 subjects |
|---|---:|---:|---:|---:|
| ADNI | 501 | 405 | 906 | 244 |
| NACC | 414 | 240 | 654 | 527 |

相对 CN vs AD：ADNI 更均衡；NACC AD 仍偏少（类不平衡需在报告中显式给出 SEN/SPE / worst-class recall）。

---

## 2. 可复现记录

- 配置：
  - 冻结：`journal_dual_shift_apis_v2_claim_mci_ad.yaml`
  - 本机：`journal_dual_shift_apis_v2_claim_mci_ad_remote.yaml`（仅路径）
- 输出根：`outputs/journal/dual_shift_apis_v2/claim_mci_ad/`
- 协议名：`apis_v2_claim_e1_mci_ad`
- 启动器：`experiments/run_apis_v2_claim_e1.py`（已支持 `claim_<task>` 输出根；`max_workers`≤2）
- 汇总：`experiments/report_apis_v2_claim_e1.py --seed-root .../claim_mci_ad/e1`

```bash
cd /zjs/AD_Project/dual_shift_github   # feature branch with this plan
# ensure dataset + scan_manifests symlinks (same as postfix remote)
export JOURNAL_PYTHON=/opt/conda/envs/cyh/bin/python
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

$JOURNAL_PYTHON -m unittest discover -s tests -p 'test_apis*.py' -v
$JOURNAL_PYTHON -m unittest discover -s tests -p 'test_claim*.py' -v

$JOURNAL_PYTHON experiments/run_apis_v2_claim_e1.py \
  --config_path journal_dual_shift_apis_v2_claim_mci_ad_remote.yaml \
  --fingerprint-only

$JOURNAL_PYTHON experiments/run_apis_v2_claim_e1.py \
  --device cuda --max-workers 2 \
  --config_path journal_dual_shift_apis_v2_claim_mci_ad_remote.yaml \
  --seeds 42,43,44,45,46

$JOURNAL_PYTHON experiments/report_apis_v2_claim_e1.py \
  --seed-root outputs/journal/dual_shift_apis_v2/claim_mci_ad/e1 \
  --seeds 42,43,44,45,46 \
  --output-dir outputs/journal/dual_shift_apis_v2/claim_mci_ad/e1
```

- 墙钟：与 CN vs AD E1 同量级，约 **60–85 h**（10 jobs × 5 变体，2 卡）
- 工作区：开跑前写 `claim_mci_ad/env_fingerprint.json`；记录 git HEAD

---

## 3. 分析与结果（预留）

| 方法 | balanced_acc | 辅助 | 判定 |
|---|---:|---:|---|
| ce_only / mixstyle / metadata / metadata_xda / apis_v2 | 待跑 | AUC F1 SEN SPE | |
| Δ(v2−mixstyle), Δ(v2−metadata_xda) + CI | 待跑 | — | Go/No-Go |

- 局限：MCI 标签跨队列异质性；NACC 无 1.5T；不得写成场强因果；不得与他机 CN vs AD 混报「v2 全面通关」

---

## 4. 建议下一步

- 建议动作：本机执行本文件 T0→T1→T2；他机 CN vs AD 互不干扰
- 固定条件：标签映射、变体、种子、hold-out、主终点、source-only 选模
- 进入条件：单测绿；指纹已写；数据 symlink 就绪；确认未误用 `.../claim/` 目录
- 禁止事项：并行再开 CN vs AD；开 MCI vs CN；三分类；Joint；看 target 调参；混表

### 开跑前检查单

- [ ] 他机 CN vs AD 仍占用其输出根；本机只用 `claim_mci_ad`
- [ ] `label_mapping: {2:0, 3:1}` 已冻结进 remote yaml
- [ ] dataset / scan_manifests 可读
- [ ] fingerprint 含 `label_mapping` 与 `output_root`
- [ ] GPU 策略：claim `max_workers=2`，不杀他人进程
