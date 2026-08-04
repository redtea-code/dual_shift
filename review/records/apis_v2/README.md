# APIS v2 实验记录

本目录只保存 APIS v2（protocol revision 2）实验记录，按运行环境继续分层：

- `common/`：不绑定单一 GPU 节点的 smoke 记录；
- `3090/`：RTX 3090 上的 CN vs AD claim E1 记录与机读表；
- `5090/`：RTX 5090 上的 MCI vs AD claim E1 记录与机读表。

APIC v3 的 image-only screening 记录位于 `../apic_v3/`。旧版 APIS 三种子与
Dual-Shift 历史记录仍保留在 `review/records/` 根目录，不视为 APIS v2 产物。
