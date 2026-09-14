"""Run DS-044 C1/C2 from frozen P0 embedding/probability arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ds044_c_mixture import json_ready, run_c_mixture_diagnostics


def _optional(data: np.lib.npyio.NpzFile, key: str):
    return data[key] if key in data.files else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with frozen P0 embeddings and subject IDs")
    parser.add_argument("--output", required=True, help="JSON diagnostics output")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-subjects-per-cluster", type=int, default=8)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        required = ["source_train_embeddings", "target_embeddings", "target_subject_ids"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise SystemExit(f"Missing NPZ arrays: {', '.join(missing)}")
        result = run_c_mixture_diagnostics(
            data["source_train_embeddings"],
            data["target_embeddings"],
            data["target_subject_ids"],
            source_train_subject_ids=_optional(data, "source_train_subject_ids"),
            target_probabilities=_optional(data, "target_probabilities"),
            target_scan_probabilities=_optional(data, "target_scan_probabilities"),
            pca_components=args.pca_components,
            n_bootstrap=args.bootstrap,
            random_state=args.seed,
            min_subjects_per_cluster=args.min_subjects_per_cluster,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_k": result["selected_k"], "target_subject_count": result["target_subject_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

