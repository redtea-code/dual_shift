# DS-043 NACC→ADNI 反向实验计划与执行记录

- 方向：`NACC_to_ADNI`
- 协议定位：unsupported-protocol stress test
- 变体：`P0`、`P0-M`、`F0`、`F1`、`F2`、`F3`、`R1`、`R2`、`R3`
- seeds：42、43、44
- target adaptation：无标签 ADNI MRI/image-only view
- 输出：`outputs/ds043_reverse/`

本实验复用 DS-043 正向实验的模型、训练预算和指标定义。所有 target label 仅在冻结模型后的最终评估阶段读取。该方向用于压力测试，不与 ADNI→NACC 主方向合并排名。

运行完整性：27/27 个变体-seed 任务生成 `report.json`、`predictions.json`，且 exit records 全部为 `exit_code=0`、`state=completed`。
