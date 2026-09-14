# 实验文档分支

本分支专门保存 `dual_shift` 项目的实验文档，包括实验计划、协议、分析、决策、结果记录和模板。代码、训练脚本、测试和数据文件保留在 `main` 分支。

## 文档目录

- `docs/plans/`：实验计划与执行计划
- `docs/results/`：实验结果记录与详细报告
- `docs/protocols/`：数据和评估协议
- `docs/analysis/`：审计、分析和方法总结
- `docs/decisions/`：数据事实与结论边界
- `docs/templates/`：实验计划和结果模板

根目录中的 `README.md` 和 `SOURCE_TEST_REPORT.md` 作为项目文档入口一并保留。

## 分支整理说明

本分支以远端 `main` 的合并基线 `606b246` 建立，保留了主线已有的全部文档内容，并补入其他实验分支中独有的计划、协议和结果报告。代码变更不在本分支范围内。

补入的内容包括 DS-042、DS-043、DS-044 文档，以及历史 `review/plans/` 计划。相同路径存在多个版本时，优先采用 `exp/ds043-run` 中较新的实验记录；未纳入与文档无关的模型、训练脚本和测试代码。

截至 2026-09-14，分支共包含 85 个 Markdown/文本文档文件。

## 使用方式

```bash
git switch docs/experiment-records
git pull --ff-only origin docs/experiment-records
```

新增实验时，建议为实验分配 `DS-xxx` 编号，并同时新增计划文件和 `docs/results/DS-xxx/README.md` 结果记录。
