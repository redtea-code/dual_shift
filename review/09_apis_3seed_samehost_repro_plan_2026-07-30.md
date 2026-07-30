# APIS-3SEED-SAMEHOST-REPRO-20260730：同机三种子 Gate A 复现

## 基本信息

- 日期：2026-07-30
- 负责人：远程 Linux 节点执行；汇总与 Gate A 判定对齐 `review/08_dual_shift_apis_3seed_analysis_2026-07-30.md`
- 状态：计划中
- Git commit：分析基线 `15aa231`（含 `08`）；执行前须记录训练代码与配置的完整 commit / SHA
- 关联文档：`review/08_dual_shift_apis_3seed_analysis_2026-07-30.md`、`review/07_apis_3seed_windows_vs_remote_compare_2026-07-30.md`、`review/05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md`、`review/01_dual_shift_next_step_decision_2026-07-29.md`、`docs/EXPERIMENT_RECORD_TEMPLATE.md`
- 关联数据文件：计划写入 `outputs/journal/dual_shift_apis_3seed_samehost/`；完成后生成 `gate_report_3seed.json`、`metrics_table_3seed.csv`（本机单一来源，不与 Windows 混表）

## 1. 实验指导与依据

- 研究问题：在**单一预注册参考主机**上，用同一代码、环境与数据清单，APIS 是否能在 ADNI↔NACC 双向、seeds 42/43/44 上全部通过正式 Gate A。
- 假设与基线：`apis_only` 相对同机 `ce_only` 满足 Gate A；`mixstyle` 仅作参照。当前 Windows 正式 Gate A = **No-Go**；Linux 仅有 43/44 且与 Windows 在 seed43 NACC→ADNI 上 ΔF1 符号相反（−0.1063 vs +0.1815），环境冲突是唯一可能改变正式决策的证据缺口。
- 实施依据：`review/08` §4「建议下一步」；`review/07` 规定 Windows/Linux 数值不可混用；`review/01` 冻结 postfix、禁止 target 调参与任务扩展。
- 实验范围：
  - 任务：CN vs AD（label 1↔0 / 3↔1）
  - 方向：ADNI→NACC、NACC→ADNI
  - Seeds：**42、43、44**（同机全量；不得只补 42 后与异机 43/44 混判）
  - 变体：`ce_only`、`mixstyle`、`apis_only`
  - 配置：冻结 `journal_dual_shift_postfix.yaml` 超参；本机仅允许路径 remap
- 判定标准（运行前锁定，与 `08`/`05` 一致）：
  - 每个 seed×direction：`APIS AUC >= CE AUC - 0.01` ∧ `APIS macro-F1 > CE macro-F1` ∧ 无 SEN/SPE collapse（阈值 <0.05）
  - **Gate A = Go** 当且仅当双方向 × 3 seeds（共 6 个 gate）全部通过
  - 任一失败 → **No-Go**，停止性能扩展
  - 停止条件：同机包完成后无论 Go/No-Go 均归档并停止；禁止为过 gate 改超参后重跑

## 2. 可复现记录

- 配置文件：
  - 冻结源：`config/journal_dual_shift_postfix.yaml`（或仓库根 `journal_dual_shift_postfix.yaml`）
  - 本机执行：`config/journal_dual_shift_postfix_remote.yaml`（**仅** `image_root` / `metadata_csv` / `scan_manifest.root` 路径；超参字节级应与冻结源一致，除路径字段外）
- 数据与划分版本：
  - `scan_manifests/ADNI_scan_manifest.csv` SHA256：`63bd665284a571f3d33362cf49630c12d1506d891484559a3375483160e626d4`
  - `scan_manifests/NACC_scan_manifest.csv` SHA256：`df99fa160a77910dc4dd5eafe4b654f1c288d91a52760fb3a4dc03eafe9fb5b1`
  - 期望规模（与既有 remote split 对齐）：ADNI 791 scans / 257 subjects；NACC 1181 scans / 960 subjects（subject-mean 聚合）
  - 每个 run 必须保留 `split_manifest.json`；三 seed 间人数应一致、清单可按 seed 不同
- 随机种子：42、43、44
- 环境与硬件（参考主机预注册 — 本 Linux 节点）：
  - Host：`an5bi4acenfa1-0`
  - OS：Linux 5.15 / glibc 2.31
  - Python：`/opt/conda/envs/cyh` 3.10.20
  - PyTorch：2.13.0+cu130；CUDA 13.0
  - GPU：6× NVIDIA GeForce RTX 5090；队列 `max-workers≤3`
  - 启动前再跑一次版本转储写入 `outputs/.../env_fingerprint.json`
- 启动命令（预注册）：

```bash
cd /zjs/AD_Project/dual_shift   # 或对齐 github 的同提交工作树
export JOURNAL_PYTHON=/opt/conda/envs/cyh/bin/python
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

# T0：指纹归档（进入条件）
$JOURNAL_PYTHON - <<'PY'
import json, platform, subprocess, hashlib, sys
from pathlib import Path
import torch
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()
root=Path('.')
payload={
  "host": platform.node(),
  "platform": platform.platform(),
  "python": sys.version,
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
  "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
  "git": subprocess.getoutput("git rev-parse HEAD 2>/dev/null || echo NA"),
  "config_sha256": {
    "postfix": sha("config/journal_dual_shift_postfix.yaml"),
    "remote": sha("config/journal_dual_shift_postfix_remote.yaml"),
  },
  "manifest_sha256": {
    "ADNI": sha("scan_manifests/ADNI_scan_manifest.csv"),
    "NACC": sha("scan_manifests/NACC_scan_manifest.csv"),
  },
}
out=Path("outputs/journal/dual_shift_apis_3seed_samehost/env_fingerprint.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out)
PY

# T1：同机全量重跑（新输出根，避免覆盖历史 Windows/旧 remote 目录）
$JOURNAL_PYTHON scripts/run_journal_queue.py \
  --stages apis_3seed_samehost \
  --device cuda --max-workers 3 \
  --plan review/09_apis_3seed_samehost_repro_plan_2026-07-30.md
# 若队列尚未注册该 stage，等价展开为 6 个 job：
# seeds 42,43,44 × {ADNI_to_NACC,NACC_to_ADNI}，variants=ce_only mixstyle apis_only
# --output-dir outputs/journal/dual_shift_apis_3seed_samehost/seed{S}/{adni_to_nacc|nacc_to_adni}
# --config_path config/journal_dual_shift_postfix_remote.yaml

# T2：优先冒烟 / 复核冲突点（可先串行跑完再跑其余）
$JOURNAL_PYTHON run_v2.py --exp journal --direction NACC_to_ADNI \
  --variants ce_only mixstyle apis_only --seed 43 --device cuda \
  --output-dir outputs/journal/dual_shift_apis_3seed_samehost/seed43/nacc_to_adni \
  --config_path config/journal_dual_shift_postfix_remote.yaml

# T3：同机 Gate A 汇总（seeds 显式 42,43,44；postfix-root 指向同机 seed42 目录布局）
$JOURNAL_PYTHON scripts/report_apis_3seed.py \
  --seeds 42,43,44 \
  --seed-root outputs/journal/dual_shift_apis_3seed_samehost \
  --postfix-root outputs/journal/dual_shift_apis_3seed_samehost/seed42 \
  --output-dir outputs/journal/dual_shift_apis_3seed_samehost
```

- 工作区状态：启动前要求与实验相关路径 clean，或将 dirty 文件列表写入 `env_fingerprint.json`；禁止在查看 target 指标后改配置。
- 产物位置：
  - 主目录：`outputs/journal/dual_shift_apis_3seed_samehost/`
  - 须上传：metrics / predictions / summary / split_manifest / gate 报告 / env_fingerprint / 队列日志摘要
  - checkpoint（`.pt`）可留本机；上传可选，但须记录路径与大小

## 3. 分析与结果

### 3.1 结果

| 方法/条件 | 主要指标 | 辅助指标 | 判定 | 备注 |
|---|---:|---:|---|---|
| 同机 ce_only（per seed×dir） | AUC / macro-F1 | Brier / ECE / SEN / SPE | 基线 | 待跑 |
| 同机 apis_only | 相对 CE 的 ΔAUC / ΔF1 | 校准与塌缩 | Gate 逐格 | 待跑 |
| 同机 Gate A（6/6） | — | — | Go / No-Go | 待跑 |
| 对照：历史 Windows / 旧 Linux | 不参与正式均值 | 仅作分歧表 | 非正式 | 见 `07`/`08` |

### 3.2 分析

- 相对基线：待同机六格填完后，按 seed 与方向报告绝对差值；**禁止**与 Windows 或旧 remote 目录做混合三种子均值。
- 异常与局限：若 seed43 NACC→ADNI 仍与 Windows 符号相反，则将「环境敏感」升级为论文机制边界表述，而不是再开调参。
- 结果结论：仅当同机 6/6 pass 才允许讨论下一阶段机制验证或任务扩展；否则维持 **No-Go**。

## 4. 建议下一步实验指导

- 建议动作：在本文件预注册的参考主机上执行 **T0→T1（或 T2 优先）→T3**；完成后按模板另写 `10_..._analysis_...md`。
- 建议依据：`08` 指出唯一可能改写正式 Gate A 的冲突是运行环境；同机全量是最小可判决实验。
- 固定条件：postfix 超参、manifest、subject-mean、CE 基线、seeds、Gate 阈值、聚合方式全部冻结。
- 进入条件：`env_fingerprint.json` 已写入且配置/manifest SHA 与本文一致；GPU 空闲且不杀他人进程；新输出根为空或显式 `--force` 并记入日志。
- 禁止事项：复现完成前不扩展 MCI/三分类、不恢复 Joint、不补 CDT 多 seed、不按 target 改 APIS/CDT 超参、不混合 Windows 与 Linux seed 结果、不把「仅 43/44 全过」写成正式 Gate A Go。

---

## 附录 A：作业矩阵（6 jobs × 3 variants）

| Job | seed | direction | 优先级 | 输出目录 |
|---|---:|---|---|---|
| J42A | 42 | ADNI→NACC | 标准 | `.../seed42/adni_to_nacc` |
| J42N | 42 | NACC→ADNI | 标准 | `.../seed42/nacc_to_adni` |
| J43A | 43 | ADNI→NACC | 标准 | `.../seed43/adni_to_nacc` |
| J43N | 43 | NACC→ADNI | **最高**（Win/Linux 冲突点） | `.../seed43/nacc_to_adni` |
| J44A | 44 | ADNI→NACC | 标准 | `.../seed44/adni_to_nacc` |
| J44N | 44 | NACC→ADNI | 标准 | `.../seed44/nacc_to_adni` |

并行：最多 3 卡；建议首批 `J43N, J42A, J42N`，随后 `J43A, J44A, J44N`。

## 附录 B：与历史产物的关系

| 目录 | 角色 |
|---|---|
| `outputs/apis_3seed/`（Windows） | 历史正式 No-Go 证据；保留，不覆盖 |
| `outputs/journal/dual_shift_apis_3seed/`（旧 remote 43/44） | 敏感性对照；**不**并入本同机 Gate A 均值 |
| `outputs/journal/dual_shift_apis_3seed_samehost/` | **本实验唯一正式汇总根** |
