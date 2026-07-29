# Git 项目协作与实验记录指南

本指南适用于 `journal_version2`。目标是：

1. 团队成员能够安全地共享、审阅和回退代码。
2. 每个实验都能追溯到代码版本、配置、数据版本和结果结论。
3. 不把大型输出、模型权重、患者数据或隐私信息塞进普通 Git 仓库。

## 1. 当前项目应当管理什么

### 放入 Git

- 源代码：`Model/`、`training/`、`utils/`、`data/`、`scripts/`、`tests/`、`experiments/`
- 配置文件：`config/`
- 项目文档：`docs/`、`review/`
- 小型结构化结果：CSV、TSV、JSON、Markdown
- 必要的小型 Excel 汇总表（可以保存，但尽量同时导出 CSV/TSV）
- 环境定义文件，例如 `requirements.txt` 或 `environment.yml`

注意：本项目的 `data/` 目前是数据加载与预处理源码，不是原始数据目录，因此应提交。`weights/` 目前主要是小型 TSV 结果表，也应提交；这个目录名容易误解，后续可考虑改名为 `results_tables/`。

### 不放入普通 Git

- `outputs/` 中的训练输出、检查点和压缩包；目前该目录约 4.5 GB
- `*.pt`、`*.pth`、`*.ckpt` 等模型文件
- 大型日志、缓存、虚拟环境和 IDE 配置
- ADNI/NACC 原始数据、患者级数据、访问密钥和密码

这些规则已写入项目根目录的 `.gitignore`。需要长期保存的大文件应放在团队服务器、对象存储或受控网盘中；在实验文档里记录稳定路径、文件大小和校验值。只有团队明确决定后才引入 Git LFS，避免普通 Git 与 LFS 混用造成困惑。

## 2. 第一次建立仓库（由一位负责人执行）

先在 GitHub、GitLab 或单位内部 Git 服务上创建一个**空的私有仓库**，不要在线生成 README 或 `.gitignore`。然后在项目根目录运行：

```powershell
git init
git branch -M main
git add .gitignore docs/GIT_PROJECT_GUIDE.md
git add Model training utils data scripts tests experiments config review weights
git add run_v2.py *.xlsx
git status
git commit -m "chore: initialize journal_version2 project"
git remote add origin <远端仓库地址>
git push -u origin main
```

在执行 `git commit` 前必须仔细检查 `git status`。确认没有患者数据、密钥、大型输出或检查点。可用下面的命令检查即将提交的文件：

```powershell
git diff --cached --stat
git diff --cached
```

二进制 Excel 不会出现在文本差异中，因此要额外核对文件名。首次提交建议由第二位成员复查后再推送。

## 3. 每位成员的首次设置

安装 Git 后设置自己的身份，只需执行一次：

```powershell
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱"
git config --global core.autocrlf true
```

克隆项目：

```powershell
git clone <远端仓库地址>
cd journal_version2
```

不要通过聊天软件互传整个项目压缩包。代码交换统一走远端仓库，大型实验产物统一走团队约定的存储位置。

## 4. 推荐的日常协作流程

`main` 始终保持可运行。每项工作使用一个短分支，不直接在 `main` 上长期开发。

```powershell
# 1. 开始工作前同步主分支
git switch main
git pull --ff-only

# 2. 创建工作分支
git switch -c exp/dual-shift-ablation

# 3. 修改后查看内容
git status
git diff

# 4. 分批提交相关改动
git add config/experiments/dual_shift.yaml experiments/dual_shift.py
git commit -m "feat: add dual-shift ablation experiment"

# 5. 推送分支
git push -u origin exp/dual-shift-ablation
```

随后在 GitHub/GitLab 创建 Merge Request 或 Pull Request，请至少一位成员审阅，通过测试后合并到 `main`。分支名建议：

- `feat/...`：代码功能
- `fix/...`：错误修复
- `exp/...`：实验与配置
- `docs/...`：文档和结果总结

一次提交只做一类事情。推荐提交信息格式：

```text
feat: add subject-level evaluation
fix: prevent patient leakage across folds
exp: record dual-shift seed 42 results
docs: summarize stage B findings
test: cover scan manifest validation
```

## 5. 实验结果如何管理

### 一次实验的最小可追溯信息

每个正式实验都要记录：

- 唯一实验 ID，例如 `2026-07-29_dual-shift_seed42`
- 实验目的和假设
- 运行代码的 Git commit ID
- 配置文件路径和关键参数
- 数据集名称、划分版本、纳排规则；不要记录患者隐私
- 随机种子、运行环境、机器或 GPU 型号
- 启动命令
- 主要指标和失败情况
- 大型产物的外部路径与校验值
- 结论和下一步

运行实验前取得代码版本：

```powershell
git rev-parse HEAD
git status --short
```

正式结果原则上应来自“工作区干净”的提交。如果 `git status --short` 有输出，说明还有未提交修改，实验记录必须说明这些修改，否则之后无法准确复现。

### 推荐目录

```text
review/journal/experiments/
  2026-07-29_dual-shift_seed42.md
results_tables/
  dual_shift_summary.tsv
```

现有结果可暂时继续放在 `review/journal/` 和 `weights/`，等团队统一迁移时再改目录，避免一次性打乱已有脚本。

### 实验记录模板

```markdown
# <实验 ID>：<简短标题>

## 目的

<要验证的假设，以及与哪个基线比较>

## 可复现信息

- 日期：YYYY-MM-DD
- 负责人：
- Git commit：`<git rev-parse HEAD 的输出>`
- 工作区状态：clean / dirty（dirty 时说明原因）
- 配置：`config/...`
- 数据与划分版本：
- 随机种子：
- 环境与硬件：
- 命令：`python ...`

## 结果

| 方法 | AUROC | Accuracy | 备注 |
|---|---:|---:|---|
| baseline | | | |
| proposed | | | |

## 产物

- 外部路径：
- 文件大小：
- SHA-256：

## 结论

<结论、局限、是否接受假设、下一步>
```

计算大型产物的 SHA-256：

```powershell
Get-FileHash <文件路径> -Algorithm SHA256
```

### Excel 的使用规则

Git 可以保存 Excel，但不能清晰比较单元格变化，而且多人同时编辑时难以合并。建议：

1. 每张关键结果表同时维护一份 CSV/TSV，Git 以文本版本作为审阅依据。
2. 同一时间只指定一人编辑汇总 Excel。
3. 不在 Excel 中嵌入患者级敏感数据。
4. 合并冲突时不要盲目选择 ours/theirs；由结果表负责人核对后重新生成。

## 6. 同步、冲突与撤销

每天开始工作时：

```powershell
git switch main
git pull --ff-only
```

将主分支更新带入自己的分支：

```powershell
git switch exp/dual-shift-ablation
git fetch origin
git merge origin/main
```

发生冲突时，Git 会在文本中标出 `<<<<<<<`、`=======`、`>>>>>>>`。与相关成员确认正确内容，编辑完成后：

```powershell
git add <已解决的文件>
git commit
```

新手阶段建议用 `merge`，暂不要求 `rebase` 或强制推送。绝不要在共享分支使用 `git push --force`。

常用的安全撤销方式：

```powershell
# 丢弃某个尚未暂存文件的本地修改（执行前确认不再需要）
git restore <文件>

# 取消暂存，但保留文件内容
git restore --staged <文件>

# 用一个新提交撤销历史提交，适合已推送的内容
git revert <commit-id>
```

不确定时先执行 `git status`，不要使用 `git reset --hard`。

## 7. 合并前检查清单

- `git status` 中没有意外文件
- 测试通过，至少运行与改动相关的测试
- 配置中没有个人绝对路径、密码或 token
- 没有患者数据、检查点、大型日志和生成输出
- 新实验记录了 commit、配置、数据划分、seed、命令和指标
- 关键表格提供 CSV/TSV 文本版本
- Pull Request 描述清楚“改了什么、为什么、如何验证”

## 8. 团队最小约定

建议团队在第一次使用 Git 的会议上确认以下规则：

1. 远端仓库为私有，成员权限按需分配。
2. `main` 禁止直接推送，必须通过 Pull/Merge Request。
3. 至少一人审阅后合并；涉及数据划分和指标计算时由两人核对。
4. 大文件存储的统一位置、命名规则、备份责任人和保留期限。
5. 结果表的唯一负责人，以及 CSV/TSV 与 Excel 的同步方式。
6. 若敏感数据曾被误提交，立即停止推送并通知负责人；仅删除当前文件不等于删除 Git 历史。

先把这套最小流程稳定执行，比一开始引入复杂分支模型更重要。
