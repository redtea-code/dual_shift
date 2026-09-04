# DualShift 实验分支规则

本仓库将 `main` 作为研究记录主线，将实验分支作为短期实现分支。这样可以让计划、报告和代码保持同一套可追溯的入口，同时避免实验分支之间互相覆盖历史记录。

## 1. `main` 保存什么

- 所有研究主地图、副地图、实验计划、数据协议、审计、结果报告和文字说明。
- 已完成实验的指标摘要、配置快照和必要的运行 provenance。
- 文档索引、README 和跨实验结论。新的 `DS-xxx` 记录必须同步更新 `docs/EXPERIMENT_INDEX.md`。

## 2. 新实验分支保存什么

新实验分支命名为 `exp/<id>-<short-name>`，默认只提交：

- `Model/**` 中的模型、损失和适配模块；
- 必要的 `training/**` 训练循环或数据流改动。

实验分支不新增或修改计划、报告、研究地图、README、结果叙述和本地路径说明。实验结束后，将实现和可复核的结果材料分别合并回 `main`；结果材料不得只留在实验分支。

## 3. 例外与合并边界

数据加载、配置、脚本和测试只有在实现确实需要时才随实验提交，并在计划或结果记录中说明原因。`outputs/` 只保留小型、可审计的摘要；大规模日志、checkpoint 和预测文件留在服务器，不提交到 Git。

合并前必须确认：

1. target label、target prediction 和 target metric 没有进入 UDA 的训练、选择或停止逻辑；
2. source/target split、seed、checkpoint 和配置可以由记录复现；
3. 报告中的数值与提交的代码、配置和摘要文件对应；
4. `git diff --check` 通过，且 staged patch 只包含本次实验的明确范围。

## 4. 当前整合分支

`codex/dualshift-consolidated` 汇总 DS-041、DS-042、DS-043 及此前 `main` 的历史文档和实现。后续实验应从更新后的 `main` 创建 `exp/<id>-<short-name>`，文档和报告直接回收到 `main`。
