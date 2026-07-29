import os
import sys
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["ANTS_RANDOM_SEED"] = "42"

FIXED_PATH = r"D:\MNI152\MNI152_T1_1mm_Brain.nii.gz"
SOURCE = r"E:\2.causal\NACC_m"
TARGET = r"E:\2.causal\NACC_m"


def _init_worker():
    """Per-worker initialisation: ignore SIGINT so only the parent handles Ctrl+C."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_subject(sub):
    """Process a single subject.  fixed is loaded inside the worker to avoid
    pickle issues with ANTsImage objects on Windows spawn."""
    import ants

    source_file = os.path.join(SOURCE, sub, "MRI_brain_mask.nii.gz")
    target_file = os.path.join(TARGET, sub, "MRI_brain_mask_mni152.nii.gz")

    if os.path.exists(target_file):
        return sub, "skipped", None

    try:
        fixed = ants.image_read(FIXED_PATH)
        moving = ants.image_read(source_file)
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyN",
        )
        ants.image_write(reg["warpedmovout"], target_file)
        return sub, "success", target_file
    except Exception:
        return sub, "error", traceback.format_exc()


def get_tasks():
    """Return subjects that still need processing."""
    tasks = []
    for sub in os.listdir(SOURCE):
        sub_dir = os.path.join(SOURCE, sub)
        if not os.path.isdir(sub_dir):
            continue
        target_file = os.path.join(TARGET, sub, "MRI_brain_mask_mni152.nii.gz")
        if not os.path.exists(target_file):
            tasks.append(sub)
    return tasks


if __name__ == "__main__":
    tasks = get_tasks()

    if not tasks:
        print("所有被试已完成，无需处理。")
        sys.exit(0)

    # Use at most 8 workers to avoid memory pressure (MRI ~200-500MB each)
    num_workers = min(cpu_count(), len(tasks), 8)
    print(f"待处理: {len(tasks)} / 并行进程数: {num_workers} / CPU核数: {cpu_count()}")

    with Pool(processes=num_workers, initializer=_init_worker) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_subject, tasks),
            total=len(tasks),
            desc="MNI152 ",
            unit="sub",
        ))

    success = sum(1 for r in results if r[1] == "success")
    skipped = sum(1 for r in results if r[1] == "skipped")
    errors  = sum(1 for r in results if r[1] == "error")

    print(f"\n===== 处理完成 =====")
    print(f"  成功: {success}")
    print(f"  跳过(已存在): {skipped}")
    print(f"  失败: {errors}")

    if errors:
        print("\n----- 错误详情 -----")
        for sub, status, detail in results:
            if status == "error":
                print(f"\n[{sub}]")
                print(detail)

