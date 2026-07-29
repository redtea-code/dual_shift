"""MONAI transform pipelines for 3D medical image preprocessing."""
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlip, RandAffine, RandGaussianNoise, RandGibbsNoise,
    ScaleIntensityd, Orientationd, ResampleToMatchd,
    EnsureTyped,
)

aal_path = r'E:\\GRETNA\\Atlas\\AAL116_1mm.nii'
template_path = r'E:\\GRETNA\\MNI152_T1_1mm_Brain.nii.gz'
target_size = (160, 160, 96)
target_spacing = (1.0, 1.0, 1.0)

register_transform = Compose([
    EnsureChannelFirstd(keys=["image", "template", 'aal']),
    Orientationd(keys=["image", 'template'], axcodes="RAS"),
    ResampleToMatchd(
        keys="image",
        key_dst="template",
        mode="bilinear"
    ),
    Resized(keys=['image'], spatial_size=target_size, mode='bilinear'),
])

register_transform2 = Compose([
    EnsureChannelFirstd(keys=["template", 'aal']),
    Orientationd(keys=["aal", "template"], axcodes="RAS"),
    ResampleToMatchd(
        keys="aal",
        key_dst="template",
        mode="nearest"
    ),
    Resized(keys=['aal'], spatial_size=target_size, mode='nearest'),
])
