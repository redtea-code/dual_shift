# APIS v2 两个中期分支综合分析报告

日期：2026-08-03  
审阅对象：`feature/apis-v2-claim-p0`、`feature/apis-v2-claim-mci-ad`  
当前集成状态：已合并至 `main`，对应合并提交 `98b9126`、`57211b9`

## 1. 结论摘要

两条分支均已完成代码审阅、配置校验和针对性测试，没有发现阻止实验继续进行的代码级问题，因此已合并到 `main`。这表示“实验协议、启动器和记录可以进入共同实验基线”，不表示 APIS v2 的性能主张已经被证明。

截至本次中期检查，跨任务证据呈现明显的任务依赖性：

- **CN vs AD**：相对 MixStyle 有小幅正向趋势，但 seed 间不稳定，且预注册的完整 5-seed 配对 bootstrap 门控尚未完成。
- **MCI vs AD**：已完成的 seed×方向比较中，APIS v2 在主终点 balanced accuracy 上均低于 MixStyle，当前不能支持正向主张。
- **metadata_xda**：多个目标域结果接近随机水平，差值很大但机制解释价值有限；必须先完成基线塌缩审计。

因此当前最合适的论文口径是：APIS v2 的有效性仍处于确认阶段，CN vs AD 值得完成全矩阵，MCI vs AD 应作为高风险任务报告，而不能把两个任务合并成“普遍优于 MixStyle”的结论。

## 2. 分支变更与合并判断

### 2.1 P0 分支

该分支新增了 Claim E1 的中期记录和综合判断，沿用协议 r2 的 subject-level hold-out、固定 split seed、source-only 选模和预注册主终点。文档明确区分了已完成结果、未完成矩阵和不可提前宣称的结论。新增内容为实验审计材料，不改变训练逻辑，合并风险低。

### 2.2 MCI vs AD 分支

该分支新增独立任务配置和输出根 `outputs/journal/dual_shift_apis_v2/claim_mci_ad`，使用 `2 -> 0 (MCI)`、`3 -> 1 (AD)` 的标签映射，避免与 CN vs AD 产物混写。启动器增加了 GPU 槽位队列、`CUDA_VISIBLE_DEVICES` 绑定、任务专属指纹字段，并补充了对应 launcher 测试。关键验证结果如下：

| 检查 | 结果 |
|---|---|
| `tests/test_apis_v2.py` | 9/9 通过 |
| `tests/test_claim_launcher.py` | 3/3 通过 |
| 三个 claim 配置校验 | 全部通过 |
| 关键文件编译检查 | 通过 |

该分支主要扩展实验隔离和可重复执行能力，不会绕过 hold-out 或改变主终点，故可合并。需要留意的是，MCI 远程计划文档中仍有 `max_workers=2` 的旧表述，而当前启动器和中期记录使用 3 个 GPU 槽位；这属于文档同步问题，不是运行逻辑阻断项，后续应统一记录。

## 3. 中期结果解读

### 3.1 CN vs AD

在已完成的 r2 子集上：

- ADNI→NACC：APIS v2 相对 MixStyle 的平均 ΔBA 约 `+0.030`，4 个 seed 中赢 3 个。
- NACC→ADNI：平均 ΔBA 约 `+0.051`，3 个已完成 seed 中赢 2 个。

这些结果支持“存在值得继续验证的信号”，但不支持“显著优于 MixStyle”。尤其是单个 seed 的反向结果说明增益并非对初始化稳定。完整 5-seed 矩阵和 subject-level 配对 bootstrap CI 是决定 E1 是否通过的必要条件。

### 3.2 MCI vs AD

在 5090 节点的中期记录中，已完成的 42 A→N、42 N→A、43 A→N 等 seed×direction 比较里，APIS v2 的主终点 BA 均低于 MixStyle。部分 AUC 结果偶有改善，但这不能抵消主终点的不利方向。当前结论应标记为“负向/未决”，继续跑完整矩阵的目的主要是确认效应是否稳定，而不是预设其会逆转。

### 3.3 metadata 基线

CN vs AD 与 MCI vs AD 记录都显示 metadata 或 metadata_xda 在部分目标域接近 BA=0.50。APIS 相对该基线的较大差值更可能反映基线实现、checkpoint 选择或特征融合失败，而非 APIS 的机制优势。应先完成 X-only、X+D、X+A、X+D+A 的 source-validation 审计，再决定是否把该轴作为正式主张证据。

## 4. 对主张的综合判断

当前证据只允许以下内部叙述：

> 在协议 r2 的不完整中期矩阵中，APIS v2 在 CN vs AD 上相对 MixStyle 出现小幅但不稳定的正向趋势；在 MCI vs AD 的已完成比较中暂未显示优势。metadata_xda 基线存在系统性塌缩，尚不能用于机制优越性结论。

以下表述目前不应写入论文或摘要：

- “APIS v2 显著优于 MixStyle”；
- “APIS v2 在不同诊断任务上普遍有效”；
- “APIS v2 相对 metadata 融合的优势证明了其因果/协议机制”。

## 5. 后续实验与审计清单

1. 完成 CN vs AD 和 MCI vs AD 的全部预注册 seed×direction×variant 矩阵，不改变模型、超参数、split 或主终点。
2. 运行统一 report gate，输出每个方向的均值、标准差、逐 seed ΔBA、subject-level 配对 bootstrap 95% CI 和胜负计数。
3. 对 metadata 基线执行 source-only 选模、标签映射、标准化、缺失值和 checkpoint 审计，并保留失败结果。
4. 统一远程计划中的并行度记载：当前 MCI 任务为 `max_workers=3`、GPU 槽位 `0,1,2`；CN vs AD 的异常退出后续仍建议单 worker 或加强进程监护。
5. 只有在 E1 主终点门控通过后，才使用 E3 场强一致性、描述子 shuffle 负对照和 E2 协议簇 hold-out 支持机制解释。

## 6. 最终审阅决定

**代码与实验基础设施：通过，已合并。**  
**APIS v2 性能主张：暂不通过，等待完整实验与基线审计。**

