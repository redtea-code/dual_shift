# 一阶段正式方案：无 ROI 模板的软解剖区域表征

版本：1.0
状态：待执行
适用项目：DualShift 期刊版方向
代码基线：`layer4_pixel` 及现有 `original_capm` 结构消融

## 1. 研究目的

当前 pixel 级调制直接作用于深层 feature map，但没有明确检验模型是否学到了稳定、可迁移的脑区级语义。第一阶段只回答一个问题：

> 在不引入 ROI 模板、配对扫描和 GRL 的条件下，将深层空间单元聚合为可学习的软区域 token，是否能提升 AD 分类在未见站点/协议上的泛化，同时不牺牲诊断判别力和人口学信息的合理保留？

本阶段是结构与表征验证，不声称发现了真实的海马、脑室等解剖 ROI，也不声称完成了图像 harmonization。

## 2. 明确排除项

以下内容不进入本阶段：

- 不要求同一受试者的两种场强或两种协议扫描；
- 不构造跨扫描样本，不做配对样本挖掘或配对 batch；
- 不使用配对一致性损失、配对 embedding 距离、配对预测差或配对一致性作为模型选择依据；
- 不使用 ROI/脑区模板作为输入、监督标签或训练先验；
- 不使用 GRL 或对抗域分类器；
- 不在本阶段引入 graph convolution。图模型只作为第二阶段退路，前提是软区域表征本身已经显示出稳定收益。

原有文档中关于配对一致性的内容保留为历史记录，但不属于本正式方案的可执行协议。

## 3. 可检验假设

### H1：空间语义聚合假设

在 GAP 之前对 `layer4` 空间单元进行软区域聚合，比直接 GAP 更能保留与诊断相关的中尺度结构。

### H2：无模板区域学习假设

仅由局部视觉特征和归一化空间坐标产生的软分配，可以形成跨样本可复用的区域 token；该 token 的稳定性不依赖预先给定的解剖模板。

### H3：跨域稳健性假设

区域化表征如果减少了对局部扫描纹理的依赖，将在未见站点、厂家、场强或协议簇上表现出更小的性能退化；该结论必须以外部 subject-level 指标和扫描参数探针共同支持。

## 4. 数据与划分协议

1. 沿用项目现有 subject-level train/validation/test 划分、任务标签映射、预处理、增强和评估脚本。
2. 同一 subject 的纵向或重复 scan 必须位于同一划分中；不得以 scan 数量代替独立样本数。
3. 训练和 checkpoint 选择只读取 source train/validation；target 只在最终评估阶段读取。
4. 主结果至少报告 `ADNI_to_NACC` 和 `NACC_to_ADNI`，并按现有 common-support/unsupported-protocol 口径分层。这里仅报告非配对的 subject-level 外部泛化，不报告任何配对一致性终点。
5. 所有变体使用同一 manifest、同一 split、同一 seed 集合和同一训练预算。建议正式结果使用 seeds `42, 43, 44`；若机器资源不足，至少完成 `42, 43` 并明确标记为最小复现实验。

## 5. 模型定义

### 5.1 输入特征

使用现有 3D ResNet backbone 的 `layer4` 输出，不改变 backbone 深度：

```text
F4 in R^(B x C x D x H x W)
```

对当前输入尺寸，实际空间尺寸由 forward 的张量动态确定；不能把 `175` 写死为通用假设。若运行配置产生 `D x H x W = 5 x 7 x 5`，则空间单元数 `N=175`。

将空间单元展平为：

```text
X = flatten(F4) in R^(B x N x C)
```

同时为每个空间单元生成固定的归一化坐标 `P in [-1, 1]^(N x 3)`。坐标只表达 feature-grid 位置，不编码 scanner、site 或 diagnosis。

### 5.2 软区域 tokenizer

设 `K` 为软区域数量，主实验取 `K=8`，敏感性实验取 `K in {4, 8, 16}`。先将视觉特征投影到 `d=128`：

```text
Z = phi(X) in R^(B x N x d)
```

由视觉内容和位置计算每个单元属于每个区域的分数。这里在区域维做
softmax，使每个空间单元只在 `K` 个区域之间分配；随后对每个区域的
权重做归一化汇聚：

```text
S[b, n, k] = MLP([Z[b, n], P[n]])
A[b, k, n] = softmax_k(S[b, n, k])
l[b, k] = sum_n A[b, k, n]
R[b, k] = (1 / l[b, k]) sum_n A[b, k, n] Z[b, n]
```

因此 `R in R^(B x K x d)` 是每个样本的软区域 token。softmax 在区域维度归一化，保证每个空间单元只路由到一组竞争的区域；再用 `l[b,k]` 把区域 token 变成加权平均，避免大区域仅因覆盖单元多而获得更大幅值。

核心代码骨架如下：

```python
class SoftRegionTokenizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_regions: int = 8):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.score = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_regions),
        )

    def forward(
        self, feat: torch.Tensor, coord: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # feat: [B, C, D, H, W], coord: [1 or B, N, 3]
        x = feat.flatten(2).transpose(1, 2)       # [B, N, C]
        z = self.proj(x)                          # [B, N, d]
        if coord.size(0) == 1:
            coord = coord.expand(z.size(0), -1, -1)
        logits = self.score(torch.cat([z, coord], dim=-1))
        assign = logits.transpose(1, 2).softmax(dim=-1)  # [B, K, N]
        regions = assign @ z                              # [B, K, d]
        return regions, assign
```

### 5.3 分类读出

主读出使用区域 token 的均值池化，并与现有分类头保持容量可比：

```text
r = mean_k R[:, k, :]
logits = head(r)
```

同时保留一个 `GAP + R` 变体：

```text
r_gap = GAP(F4)
logits = head(concat(r_gap, mean_k R[:, k, :]))
```

拼接后的 head 使用参数量匹配的线性层/MLP；不能把额外参数量带来的收益误认为区域建模收益。若与当前 `original_capm` 结合，则先执行现有 CAPM，再从调制后的 `layer4` 产生 `R`，并保持 CAPM 的其他实现不变。

## 6. 训练目标

本阶段只使用单 scan 的监督分类和表征正则：

```text
L = L_cls + lambda_rec L_rec + lambda_bal L_bal + lambda_smooth L_smooth
```

其中：

### 6.1 分类损失

沿用当前任务的交叉熵、类别权重和 collapse guard，不修改标签协议：

```text
L_cls = CrossEntropy(logits, y)
```

### 6.2 特征重构损失

用区域 token 重构投影后的空间特征，防止所有信息被少数 token 吞掉：

```text
X_hat = A^T R
L_rec = mean(||normalize(X_hat) - normalize(Z)||_2^2)
```

实现时可在 `X_hat` 后接一个线性 decoder 以匹配维度；decoder 只用于训练约束，不参与最终分类。

### 6.3 区域负载平衡

令每个区域的总空间负载为 `l_k = sum_n A[:, k, n]`，再归一化为
`q_k = l_k / sum_j l_j`，用均匀分布约束避免 token 塌缩：

```text
L_bal = KL(q || Uniform(K))
```

该项只防止空区域，不解释为真实解剖区域监督。

### 6.4 空间平滑

在三维 feature grid 的 6-neighborhood 边集 `E` 上约束相邻单元的分配不要剧烈跳变：

```text
L_smooth = mean_(i,j) in E ||A[:, :, i] - A[:, :, j]||_1
```

初始配置建议 `lambda_rec=0.1`、`lambda_bal=0.01`、`lambda_smooth=0.01`，只允许在 source validation 上从预注册的小网格中选择；不得使用 target 指标调参。

## 7. 训练策略

### 主协议：冻结 backbone

从现有 source-only checkpoint 加载 backbone，冻结 `conv1`、`layer1`--`layer4`，只训练 tokenizer、可选 decoder 和分类头。这样可以把第一阶段的因果问题限定为“区域化读出是否有效”，避免 backbone 重新训练掩盖结构差异。

### 次协议：联合微调

只有主协议完成且没有工程问题时，才在相同 split/seed 下增加一个联合微调版本：tokenizer/head 使用正常学习率，backbone 使用其十分之一的学习率。该版本是优化稳定性检查，不得替代冻结 backbone 的主结论。

每个变体记录代码 commit、resolved YAML、manifest hash、split hash、seed、epoch、参数量和最佳 source-validation checkpoint。

## 8. 最小实验矩阵

| 编号 | 变体 | 目的 |
|---|---|---|
| B0 | 现有 `image_only`/GAP | 全局池化基线 |
| B1 | 现有 `original_capm` | 当前表格条件化基线 |
| B2 | `SoftRegion-R` | 仅使用软区域 token 的结构增益 |
| B3 | `original_capm + SoftRegion-R` | 一阶段主模型 |
| B4 | B3 去掉坐标 `P` | 验证位置编码是否必要 |
| B5 | B3 去掉 `L_rec + L_bal + L_smooth` | 验证收益是否来自正则化而非容量 |

B0--B5 必须使用相同的 backbone 深度、输入、增强、训练轮数、优化器和分类头预算。`layer3_patch2`、`layer4_pixel`、`layer5_pixel` 的完整 scale-table 矩阵继续作为已有结构消融；本方案主模型固定在 `layer4`，不把不同尺度与新区域机制同时改变。

## 9. 评价指标与审计

### 9.1 主要终点

- 外部 subject-level balanced accuracy；
- AUROC、macro-F1、sensitivity、specificity；
- source-to-target 性能差值及按站点/厂家/场强/协议簇分层的结果；
- 3 个 seed 的均值、标准差和 subject-level bootstrap 置信区间。

### 9.2 机制终点

冻结最终 representation 后分别训练轻量 probe：

- scanner/site/field-strength probe：衡量扫描相关信息是否过强；
- age/sex probe：检查人口学信息是否被不必要地抹除；
- diagnosis probe：确认表征仍保留诊断信号。

probe 只能在 probe-train split 上训练，不能读取 target 标签。扫描 probe 的下降不能单独构成成功条件，必须与诊断性能和人口学 probe 一起解释。

### 9.3 区域表征审计

- `q_k` 的负载分布和熵，检查区域是否塌缩；
- 每个 seed 的分配图和 token 统计；
- 跨 seed 用 Hungarian matching 对齐 token 后，报告 assignment 相似度；
- 在不同站点/协议簇分别汇总区域分配，检查是否出现明显的 scanner-specific token；
- 若本地存在独立 atlas，只能在训练后做重叠分析，不能把 atlas 标签加入训练。

任何可视化都只称为 learned soft regions 或 anatomy-sensitive spatial representation，不称为经过验证的解剖 ROI 或病灶定位。

## 10. 成功、失败与停止规则

### 成功标准

B3 在至少两个 seed 和至少一个未见站点/协议簇测试上相对 B0/B1：

1. 诊断外部指标方向一致提升，且 source validation 没有类别塌缩；
2. scanner/site probe 不升高，或在诊断性能不下降的情况下下降；
3. age/sex probe 与 B0/B1 相比没有明显崩溃；
4. 区域负载不塌缩，跨 seed 分配具有可重复性。

不满足上述组合条件时，只能结论为“区域 tokenizer 未显示稳健增益”，不能用单一 source accuracy 或单张热图支持方法主张。

### 立即停止条件

- 出现 subject 跨 split 泄漏；
- target 被用于 checkpoint、超参数或阈值选择；
- `A` 的大多数负载集中到一个 token，且 balance/初始化修复无效；
- B3 的训练不稳定、类别塌缩或明显超过容量匹配预算；
- 代码、配置、manifest 或 split hash 不一致。

## 11. 交付物

每个变体至少交付：

```text
resolved_config.yaml
git_commit.txt
manifest_hashes.json
split_manifest.json
summary.json
metrics_subject_level.csv
probe_metrics.json
region_assignment_audit.json
region_assignment_examples/        # 仅保存可审计的派生可视化
```

主控汇总表必须同时列出 B0--B5、训练 seed、外部诊断指标、scan probe、demographic probe、区域塌缩检查和失败原因。

## 12. 阶段结论门

只有当 B3 通过第 10 节的组合标准，才进入第二阶段的 graph reasoning：将 `K` 个软区域 token 视为节点，并比较固定邻接、内容相似邻接和可学习邻接。若 B3 未通过，则优先回到 `K`、坐标编码和正则项的最小消融，不直接引入更复杂的 GNN。
