# APIC-V3_2-IMPL-REVIEW：模型原型与研究计划一致性审阅

## 基本信息

- 日期：2026-08-04
- 审阅对象：`Model/APIC_V3_2_MODEL_DESIGN_DRAFT.md` 与 APIC v3_2-A 原型代码
- 审阅结论：**有条件接收为机制开发原型；不批准 revision 4 正式实验**
- 关联计划：`review/plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`
- 关联代码：`Model/dual_shift/apis_v3_2.py`

## 1. Findings

### P0-1：BN 对照尚未复用同批 clean moments

当前 shifted forward 将 BN 临时切换为 eval，因此不再更新 running statistics，但它使用历史
running mean/variance；clean forward 在训练态使用当前 batch moments。即使干预为零，两条路径
也可能因 BN 统计来源不同而不相等。这不满足设计第 8 节的配对反事实要求。

关闭条件：backbone 显式缓存 clean forward 每层 BN batch mean/variance，shifted continuation
复用同一组 stop-gradient moments；加入零干预逐层 feature/logit 一致性测试和 BN update counter。

### P0-2：dropout 与 unsupported 严格回退尚未成立

clean/shifted 分类头分别调用 `nn.Dropout`，训练时使用不同 mask；unsupported 样本虽然浅层
shift 为零，最终 logits 仍可能不同。设计要求共享 dropout mask，并让 unsupported 最终
embedding/logits 直接复用 clean 值。

关闭条件：显式生成并复用 dropout mask；对 `valid_mask=false` 行执行 embedding/logit 覆盖；
加入逐样本严格回退测试。

### P0-3：损失函数与预注册定义不一致

当前实现对聚合后的 clean CE 和 supported shifted CE 各取一半，shifted 项按 `sum(mask)` 归一；
设计要求逐样本保持总分类权重为 1，并让所有辅助项按整个 batch 大小归一。当前 RMS band 仍是
可优化损失，而设计已将其冻结为只读审计量；现有 `style_target_error` 还是 detach 后的残差幅度，
不能向浅层目标提供梯度。

关闭条件：实现逐样本分类权重公式；辅助项改为 `B^-1 sum(m_i loss_i)`；删除 RMS 优化项；
实现 mean/log-std 的逐通道 Huber style loss，并验证其梯度非零。

### P0-4：prototype fit/calibration 数据协议未实现

当前 bank 使用 warm-up B0 扫描描述符直接拟合 PCA、k-means 和 support radius；未建立
subject-disjoint 的 `mechanism_fit` / `mechanism_calibration`，未使用 subject-balanced 权重，
support radius 也来自拟合集本身。小簇通过移动样本修复，与设计要求的“初始化失败即停止”相反。

关闭条件：生成并 hash subject-level 机制拆分；仅用 fit 拟合 median/IQR、PCA、weighted k-means
与目标统计；仅用 calibration 冻结半径和强度；不合格 restart 明确失败。

### P1-1：描述符、门控和目标选择未完全冻结

原型代码仍包含 low/high-frequency energy，而最新设计冻结为 mean/log-std；gate 使用 softmax
confidence，不是预注册的簇内相对距离公式；目标固定选择最近替代原型，未执行
`delta_min/delta_max` 与无状态 subject/scan hash 采样。

关闭条件：按设计收敛描述符；实现固定 gate；保存原型间合法性矩阵；目标选择对 DataLoader
顺序、worker 和设备数保持确定性。

### P1-2：checkpoint selector 与 mechanism guard 未实现

原型沿用现有 composite selector，尚未实现 3-epoch EMA subject-level BA 排序，也没有在唯一
performance-selected checkpoint 上执行 mechanism guard。当前日志不足以证明 Gate M0。

关闭条件：三种 revision 4 方法共享新 performance selector；APIC 只在该 checkpoint 上判机制；
无合格机制时直接失败而不重选。

### P1-3：审计输出仍不完整

已有有效 slot、最大 slot share、gate、平均 RMS 和 style 量，但缺少 subject occupancy、分层
supported fraction、逐层 RMS 分位数/band hit rate、失败原因计数、BN moment 配对标识和
subject-level assignment-label NMI。

关闭条件：逐项实现设计第 12 节输出，并以结构化 JSON/CSV 测试字段和统计口径。

## 2. 已确认可保留的实现

- 独立命名和输出树：`APIC_v3_2` / `v3_2_balanced_style_memory` / `apic_v3_2_x`；
- image-only 模态隔离，不消费 acquisition metadata；
- warm-up 后 deep-copy style teacher，teacher 参数冻结且强制保持 eval；
- prototype、PCA、teacher 状态进入模型 state dict；
- layer1/layer2 使用相对原型统计迁移，并设置 actual relative RMS 上界；
- unsupported 浅层 shift 为零；
- shifted branch 不二次更新 BN running statistics；
- 独立 CN/MCI 配置、launcher、诊断 `--variant apic_v3_2_x` 接口；
- 原 APIC v3 变体与历史结果目录未被覆盖。

## 3. 上传与运行决策

本轮允许上传研究计划、设计文档、审阅报告、prototype 代码、配置和接口，目的为协作开发与
可追溯审阅。上传不表示 Gate M0 已通过，也不表示正式实验已获准。

两个配置必须保持：

```yaml
implementation_status: prototype_not_protocol_compliant
formal_run_allowed: false
```

launcher 默认只允许 `--fingerprint-only`；非主张开发运行必须显式提供
`--allow-prototype-run`。关闭全部 P0/P1 后，应新增复审记录，而不是覆盖本文。

## 4. 验证记录

- APIC v3_2 聚焦测试：通过；
- 原 APIC v3 回归测试：通过；
- 新增 Python 文件：`py_compile` 通过；
- CN/MCI revision 4 YAML：配置校验通过；
- 全量 pytest：被当前 `segment` 环境中损坏的 SciPy 安装阻断，错误为
  `ModuleNotFoundError: scipy._lib._util`，不计为模型通过。
