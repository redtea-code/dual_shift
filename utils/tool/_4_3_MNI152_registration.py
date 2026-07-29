"""
_4_3_MNI152_registration.py
==========================
将 MRI_N4_brain.nii.gz 配准到 MNI152 brain 模板，并将同一套 transforms
应用到 native brain mask。

核心改进（相比 _4_2）:
  - 用 MRI_N4_brain 估计配准（而非直接用 mask 配准）
  - 图像: Linear 插值  |  mask: NearestNeighbor 插值
  - mask 变换后重新二值化（threshold 0.5）
  - 同时输出配准后图像和配准后 mask，确保两者共享同一套 transforms

用法:
    python _4_3_MNI152_registration.py

可配置项（脚本顶部）:
    FIXED_PATH : MNI 模板路径
    SOURCE     : 源数据目录（按被试组织的子目录）
    TARGET     : 输出目录（通常与 SOURCE 相同，in-place 输出）
"""

import os
import sys
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["ANTS_RANDOM_SEED"] = "42"

# ========== 可配置参数 ==========
FIXED_PATH = r"D:\MNI152\MNI152_T1_1mm_Brain.nii.gz"
SOURCE = r"D:\ADNI_dataset\Gmamba\ADNI\ADNI_pre"
TARGET = r"D:\ADNI_dataset\Gmamba\ADNI\ADNI_pre"
#D:\ADNI_dataset\Gmamba\NACC\NACC_pre
# 输入文件命名（每个被试目录下）
MOVING_IMG_NAME = "MRI_brain.nii.gz"     # N4 校正后的去颅骨图像
MOVING_MASK_NAME = "MRI_brain_mask.nii.gz"  # SynthStrip 输出的 native brain mask

# 输出文件命名（写入 TARGET/sub/）
OUTPUT_IMG_NAME = "MRI_brain_mni152.nii.gz"
OUTPUT_MASK_NAME = "MRI_brain_mask_mni152.nii.gz"

# 配准参数
REG_TYPE = "SyN"                # 配准变换类型

# 并行参数
MAX_WORKERS = 8  # 最多并行进程数（MRI ~200-500MB/进程）


def _init_worker():
    """Per-worker initialisation: ignore SIGINT so only the parent handles Ctrl+C."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_subject(sub):
    """对单个被试执行 MNI152 配准。

    步骤:
      1. 读取 fixed template (MNI152 brain)
      2. 读取 moving image (MRI_N4_brain.nii.gz)
      3. 读取 moving mask (MRI_brain_mask.nii.gz)
      4. 用 moving image 估计配准（SyN 或用户指定）
      5. 将同一套 fwdtransforms 应用到 image（linear 插值）和 mask（nearestNeighbor 插值）
      6. mask 重新二值化（threshold 0.5）
      7. 写入输出文件
    """
    import ants

    moving_img_path = os.path.join(SOURCE, sub, MOVING_IMG_NAME)
    moving_mask_path = os.path.join(SOURCE, sub, MOVING_MASK_NAME)
    target_img_path = os.path.join(TARGET, sub, OUTPUT_IMG_NAME)
    target_mask_path = os.path.join(TARGET, sub, OUTPUT_MASK_NAME)

    # 跳过已完成的被试（两个输出文件都存在才视为完成）
    if os.path.exists(target_img_path) and os.path.exists(target_mask_path):
        return sub, "skipped", None

    # 输入文件检查
    for p, label in [
        (moving_img_path, "移动图像"),
        (moving_mask_path, "mask"),
    ]:
        if not os.path.exists(p):
            return sub, "missing_input", f"缺少输入文件: {label} -> {p}"

    try:
        fixed = ants.image_read(FIXED_PATH)
        moving_img = ants.image_read(moving_img_path)
        moving_mask = ants.image_read(moving_mask_path)

        # ---- 配准：用 MRI_N4_brain 图像估计形变场 ----
        reg = ants.registration(
            fixed=fixed,
            moving=moving_img,
            type_of_transform=REG_TYPE,
        )

        # ---- 图像 -> MNI（linear 插值）----
        img_mni = ants.apply_transforms(
            fixed=fixed,
            moving=moving_img,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )

        # ---- mask -> MNI（nearestNeighbor 插值）----
        mask_mni = ants.apply_transforms(
            fixed=fixed,
            moving=moving_mask,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )

        # ---- mask 重新二值化 ----
        # nearestNeighbor 理论上不应产生中间值，但为避免边缘效应，再做一次 threshold
        mask_mni = ants.threshold_image(mask_mni, 0.5, 1e9, 1, 0)

        # ---- 写入 ----
        os.makedirs(os.path.join(TARGET, sub), exist_ok=True)
        ants.image_write(img_mni, target_img_path)
        ants.image_write(mask_mni, target_mask_path)

        return sub, "success", (target_img_path, target_mask_path)

    except Exception:
        return sub, "error", traceback.format_exc()


def get_tasks():
    """扫描源目录，返回待处理的被试列表。"""
    tasks = []
    for sub in os.listdir(SOURCE):
        sub_dir = os.path.join(SOURCE, sub)
        if not os.path.isdir(sub_dir):
            continue
        target_img = os.path.join(TARGET, sub, OUTPUT_IMG_NAME)
        target_mask = os.path.join(TARGET, sub, OUTPUT_MASK_NAME)
        if not (os.path.exists(target_img) and os.path.exists(target_mask)):
            tasks.append(sub)
    return tasks


if __name__ == "__main__":
    print(f"FIXED 模板: {FIXED_PATH}")
    print(f"输入目录:   {SOURCE}")
    print(f"输出目录:   {TARGET}")
    print(f"移动图像:   {MOVING_IMG_NAME}")
    print(f"移动 mask:  {MOVING_MASK_NAME}")
    print(f"配准方法:   {REG_TYPE}")
    print()

    tasks = get_tasks()

    if not tasks:
        print("所有被试已完成，无需处理。")
        sys.exit(0)

    num_workers = min(MAX_WORKERS, cpu_count(), len(tasks))
    print(f"待处理: {len(tasks)} / 并行进程数: {num_workers} / CPU 核数: {cpu_count()}")

    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_subject, tasks),
            total=len(tasks),
            desc="MNI152 配准",
            unit="sub",
        ))

    success = sum(1 for r in results if r[1] == "success")
    skipped = sum(1 for r in results if r[1] == "skipped")
    missing = sum(1 for r in results if r[1] == "missing_input")
    errors = sum(1 for r in results if r[1] == "error")

    print(f"\n===== 处理完成 =====")
    print(f"  成功:         {success}")
    print(f"  跳过(已存在):  {skipped}")
    print(f"  缺少输入文件:  {missing}")
    print(f"  失败:         {errors}")

    for status_name, status_label in [
        ("missing_input", "缺少输入文件"),
        ("error", "错误"),
    ]:
        if any(r[1] == status_name for r in results):
            print(f"\n----- {status_label}详情 -----")
            for sub, status, detail in results:
                if status == status_name:
                    print(f"\n[{sub}]")
                    print(detail)
