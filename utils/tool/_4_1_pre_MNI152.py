import os

import ants
from tqdm import tqdm

fixed = ants.image_read(r"D:\MNI152\MNI152_T1_1mm_Brain.nii.gz")
source = r'D:\ADNI_dataset\Causal\causal_dataset1\ADNI_pre'
target = r'D:\ADNI_dataset\Causal\causal_dataset1\ADNI_pre'
for sub in tqdm(reversed(os.listdir(source))):
    if os.path.exists(os.path.join(target,sub,"MRI_brain_mni152.nii.gz")):
        print(f"{sub} has been processed")
        continue
    source_file = os.path.join(source,sub,"MRI_brain.nii.gz")
    target_file = os.path.join(target,sub,"MRI_brain_mni152.nii.gz")
    moving = ants.image_read(source_file)
    reg = ants.registration(
    fixed=fixed,
    moving=moving,
    type_of_transform="SyN"
)
    ants.image_write(reg["warpedmovout"], target_file)
    print(f"success save {target_file}")
