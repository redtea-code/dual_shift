# DS-041 Source Branch Resolution

- Source branch: `codex/ds041-capm-residual-distribution`
- Source commit: `d928ac6`
- Resolution: the DS-041 model files were already included in `main` through `codex/dualshift-consolidated` (`606b246`).
- Conflict: the source branch had an older `Model/ablation/__init__.py` that omitted the DS-042 exports. The consolidated main version is retained.

No DS-041 implementation was discarded; the branch is treated as superseded rather than merged wholesale.

