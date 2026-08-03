#!/usr/bin/env python3
"""Build full MCI-AD E1 metrics main table under review/records/5090/."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "outputs/journal/dual_shift_apis_v2/claim_mci_ad/e1"
OUT_DIR = Path(__file__).resolve().parent

SEEDS = [42, 43, 44, 45, 46]
DIRS = [("adni_to_nacc", "ADNI_to_NACC"), ("nacc_to_adni", "NACC_to_ADNI")]
VARIANTS = ["ce_only", "mixstyle", "metadata", "metadata_xda", "apis_v2"]
METRIC_KEYS = [
    "balanced_accuracy",
    "auc",
    "macro_f1",
    "accuracy",
    "sensitivity",
    "specificity",
    "brier",
    "ece",
    "worst_group_auc",
    "group_gap",
    "n",
    "n_subjects",
]


def extract_block(block: dict) -> dict:
    if not isinstance(block, dict):
        return {}
    m = block.get("metrics") or {}
    row = {k: m.get(k) for k in METRIC_KEYS}
    row["loss"] = block.get("loss")
    row["aggregation"] = m.get("aggregation")
    row["label_conflict"] = m.get("label_conflict")
    ci = block.get("bootstrap_ci") or {}
    for k in METRIC_KEYS:
        if k in {"n", "n_subjects"}:
            continue
        c = ci.get(k)
        if isinstance(c, dict):
            row[f"{k}_ci_low"] = c.get("low", c.get("ci_low", c.get("lower")))
            row[f"{k}_ci_high"] = c.get("high", c.get("ci_high", c.get("upper")))
            row[f"{k}_ci_mean"] = c.get("mean")
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            row[f"{k}_ci_low"] = c[0]
            row[f"{k}_ci_high"] = c[1]
    by_fs = block.get("metrics_by_field_strength") or {}
    for fs, fm in by_fs.items():
        if not isinstance(fm, dict):
            continue
        fs_key = str(fs).replace(" ", "")
        for k in METRIC_KEYS:
            row[f"fs_{fs_key}_{k}"] = fm.get(k)
    return row


def fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.6f}"
    return str(v)


def main() -> None:
    rows: list[dict] = []
    for seed in SEEDS:
        for slug, direction in DIRS:
            for variant in VARIANTS:
                path = ROOT / f"seed{seed}" / slug / variant / "journal_metrics.json"
                base = {
                    "task": "MCI_vs_AD",
                    "seed": seed,
                    "direction": direction,
                    "direction_slug": slug,
                    "variant": variant,
                    "present": path.exists(),
                    "metrics_path": str(path.relative_to(PROJECT)).replace("\\", "/")
                    if path.exists()
                    else "",
                }
                if not path.exists():
                    rows.append(base)
                    continue
                d = json.loads(path.read_text(encoding="utf-8"))
                base.update(
                    {
                        "claim_protocol": d.get("claim_protocol"),
                        "claim_protocol_revision": d.get("claim_protocol_revision"),
                        "split_seed": d.get("split_seed"),
                        "training_seed": d.get("training_seed"),
                        "config_hash": d.get("config_hash"),
                        "variant_display_name": d.get("variant_display_name"),
                    }
                )
                for split_name in ("target", "source_val", "source_test"):
                    block = extract_block(d.get(split_name) or {})
                    for k, v in block.items():
                        base[f"{split_name}__{k}"] = v
                rows.append(base)

    fixed = [
        "task",
        "seed",
        "direction",
        "direction_slug",
        "variant",
        "present",
        "claim_protocol",
        "claim_protocol_revision",
        "split_seed",
        "training_seed",
        "config_hash",
        "variant_display_name",
        "metrics_path",
    ]
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    def sort_key(k: str):
        if k in fixed:
            return (0, fixed.index(k))
        if k.startswith("target__"):
            return (1, k)
        if k.startswith("source_val__"):
            return (2, k)
        if k.startswith("source_test__"):
            return (3, k)
        return (4, k)

    cols = sorted(all_keys, key=sort_key)

    csv_path = OUT_DIR / "metrics_table_mci_ad_e1_main.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    target_cols = [
        "balanced_accuracy",
        "auc",
        "macro_f1",
        "accuracy",
        "sensitivity",
        "specificity",
        "brier",
        "ece",
        "worst_group_auc",
        "group_gap",
        "n",
        "n_subjects",
        "loss",
    ]

    lines: list[str] = []
    lines.append("# APIS v2 claim E1 — MCI vs AD 主表（全部实验 × 全部指标）")
    lines.append("")
    lines.append("- 日期：2026-08-03")
    lines.append("- 节点：RTX 5090（`an5bi4acenfa1-0`）")
    lines.append("- 任务：MCI vs AD（label `2→0`, `3→1`）")
    lines.append(
        "- 范围：seeds 42–46 × 双向 × `{ce_only, mixstyle, metadata, metadata_xda, apis_v2}`"
    )
    lines.append(
        "- 完整宽表（含 source_val / source_test / field-strength / bootstrap CI）："
        "`metrics_table_mci_ad_e1_main.csv`"
    )
    lines.append(
        "- 说明：矩阵预先铺满 50 行；未完成格子 `present` 为空，指标留空。"
        "本表为快照，非正式 Gate 汇总。"
    )
    lines.append("")
    n_present = sum(1 for r in rows if r.get("present"))
    lines.append(f"- 已产出 metrics：`{n_present}` / `{len(rows)}`")
    lines.append("")

    def emit_split(split: str, title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        header = ["seed", "direction", "variant", "present"] + target_cols
        lines.append("| " + " | ".join(header) + " |")
        align = ["---", "---", "---", ":---:"] + ["---:"] * len(target_cols)
        lines.append("| " + " | ".join(align) + " |")
        for r in rows:
            cells = [
                str(r["seed"]),
                r["direction"],
                r["variant"],
                "Y" if r.get("present") else "",
            ]
            for k in target_cols:
                cells.append(fmt(r.get(f"{split}__{k}")))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    emit_split("target", "主表 A：target（subject-level）")
    emit_split("source_val", "主表 B：source_val")
    emit_split("source_test", "主表 C：source_test")

    # field-strength long table
    fs_entries: list[tuple[str, str, str, str, dict]] = []
    for r in rows:
        if not r.get("present"):
            continue
        # gather fs ids from keys
        fs_ids = sorted(
            {
                k[len("target__fs_") :].split("_", 1)[0]
                for k in r
                if k.startswith("target__fs_")
            }
        )
        for fs in fs_ids:
            block = {k: r.get(f"target__fs_{fs}_{k}") for k in METRIC_KEYS}
            if any(v is not None and v != "" for v in block.values()):
                fs_entries.append(
                    (str(r["seed"]), r["direction"], r["variant"], fs, block)
                )

    lines.append("## 主表 D：target 场强分层（有值则列）")
    lines.append("")
    if not fs_entries:
        lines.append("（当前快照无场强分层字段或均为空）")
        lines.append("")
    else:
        header = ["seed", "direction", "variant", "field_strength"] + METRIC_KEYS
        lines.append("| " + " | ".join(header) + " |")
        align = ["---", "---", "---", "---"] + ["---:"] * len(METRIC_KEYS)
        lines.append("| " + " | ".join(align) + " |")
        for seed, direction, variant, fs, block in fs_entries:
            cells = [seed, direction, variant, fs] + [fmt(block.get(k)) for k in METRIC_KEYS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    md_path = OUT_DIR / "18_apis_v2_claim_mci_ad_e1_metrics_main_2026-08-03.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {csv_path} cols={len(cols)} rows={len(rows)} present={n_present}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
