# APIC v3_2 M0 尺度修复与验收记录

## 修复边界

本次修复只处理 2026-08-05 诊断确认的机制阻断和与设计草案不一致的实现：

- target prototype 合法性从 PCA 原始距离改为按两端 calibration radius 归一化的相对距离；
- 删除 k-means 不足簇的样本搬运；每个 slot 必须同时具有 fit 和 calibration 支持；
- 有 subject ID 的运行缺失 mechanism_calibration 时严格失败；
- revision-4 分类损失不再重复加入 clean CE；shifted feature consistency 以整批而非支持子批归一化；
- checkpoint history 如实命名为 `val_selector_ema_ba`。

`formal_run_allowed` 保持 `false`。本修改不构成 E3 结果，也不授权扩大 seed 或启动 X+D。

## M0 完成条件

在 E1/E2 输出满足以下条件前，不得解除正式运行阻断：

1. B0 的每一个 K=4 slot 具有最小 fit 和 calibration subject 支持；任何缺失立即停止。
2. 每个有效 source slot 至少有一个相对距离位于冻结 band 的替代 prototype。
3. E2 四个单元中 supported fraction 和 nonzero realized RMS fraction 均非零，并报告 layer1/layer2 RMS band hit rate。
4. 不支持样本的 shifted logits/embedding 严格回退到 clean；shifted BN 不更新 running statistics。
5. checkpoint reload 的 clean 概率与正式导出预测在 `1e-5` 内一致。
6. layer-2 诊断覆盖一个成功和一个失败单元，并记录 clean/shift BA、flip、JS、slot counts 和 assignment-label NMI。

## 本地代码验收

`tests/test_apis_v3_2.py` 覆盖相对尺度不变性、真实替代 target、空 calibration 严格失败、零支持损失退化和 revision-4 分类/辅助损失权重。GPU E1/E2 仍是后续机制验收，而非本地单测可替代的证据。
