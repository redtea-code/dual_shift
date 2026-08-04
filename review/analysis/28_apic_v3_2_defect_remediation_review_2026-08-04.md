# APIC v3_2 缺陷修复复审

## 结论

本次修订关闭了上一份实现复审中的四项 P0 语义缺陷，并补齐了主要 P1 接口。代码仍保持 `prototype_not_protocol_compliant`，因此没有解除正式实验闸门。

## 已修复

1. **配对 BN moments**：`DualShiftBackbone` 在 clean pass 捕获每个 BatchNorm 的 batch mean/variance，shifted pass 用 `F.batch_norm` 重放同一 moments；shifted pass 不更新 running statistics。
2. **共享随机性与严格回退**：APIC v3_2 双路径保存并恢复 dropout RNG 状态；unsupported 行在最终 logits 与 embedding 上直接使用 clean 输出。
3. **损失归一化**：clean/shifted 分类损失按完整 batch 的 subject/sample 权重归一化，unsupported 样本不被删除；style target error 保留梯度；RMS band 改为审计量，不进入优化目标。
4. **source-only fit/calibration**：style descriptor 仅使用 channel mean/log-std；subject hash 固定 80/20 拆分 mechanism_fit 与 mechanism_calibration，拟合按 subject 聚合，校准集决定 support radius；fit 支持不足时 strict finalize 直接失败。
5. **确定性门控**：target slot 使用 sample-id hash 做稳定选择，并受 `delta_min/delta_max` 约束；gate 由 support 距离映射到预注册 `[g_min,g_max]`。
6. **性能选择器**：revision-4 配置使用统一的 subject-level balanced-accuracy 三 epoch EMA；机制审计不参与重新选 checkpoint。

## 复核证据

- `python -m py_compile`：通过。
- `tests/test_apis_v3_2.py tests/test_apic_v3_screening.py tests/test_apis_v3_style_memory.py`：`12 passed`。
- `run_apic_v3_2_screening.py --fingerprint-only`：通过。

## 仍需保留的限制

真实 GPU 数据运行仍需先完成 E0/E1 的 subject-disjoint mechanism smoke，并生成冻结的 calibration grid、manifest hash 与 checkpoint mechanism guard。当前配置仍禁止无显式原型标志的正式 run；这属于协议保护，不是测试失败。
