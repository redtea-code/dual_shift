import torch

from Model.dual_shift.apis_v3_2 import FixedStyleBankAPICV32
from Model.dual_shift.backbone import DualShiftBackbone
from training.dual_shift_loop import phase_schedule


def _module():
    return FixedStyleBankAPICV32(
        layer1_channels=2,
        layer2_channels=4,
        style_dim=4,
        memory_size=2,
        min_cluster_count=1,
        rms_min=0.001,
        rms_max=0.05,
    )


def test_v3_2_bank_freezes_and_produces_bounded_supported_shift():
    torch.manual_seed(7)
    module = _module()
    image = torch.randn(8, 1, 8, 8, 8)
    layer1 = torch.randn(8, 2, 4, 4, 4)
    layer2 = torch.randn(8, 4, 2, 2, 2)
    module.observe_source(layer1, layer2)
    module.finalize_style_bank()
    assert module.finalized
    module.set_alpha(0.25)
    condition, valid = module.prepare_style_condition(image, layer1, layer2)
    shift1, shift2 = module.make_shift_fns(condition, valid_mask=valid)
    shifted1 = shift1(layer1)
    shifted2 = shift2(layer2)
    assert shifted1.shape == layer1.shape
    assert shifted2.shape == layer2.shape
    assert torch.isfinite(shifted1).all()
    assert torch.isfinite(shifted2).all()
    relative = (shifted2 - layer2).flatten(1).square().mean(1).sqrt()
    denominator = layer2.flatten(1).square().mean(1).sqrt().clamp_min(1e-6)
    assert torch.all(relative / denominator <= 0.05 + 1e-5)


def test_v3_2_teacher_stays_eval_when_parent_enters_train_mode():
    torch.manual_seed(3)
    backbone = DualShiftBackbone(layers=(1, 1, 1, 1), base_channels=2)
    module = _module()
    module.freeze_teacher(backbone)
    module.train()
    assert module.teacher is not None
    assert module.teacher.training is False
    bn_layers = [item for item in module.teacher.modules() if isinstance(item, torch.nn.modules.batchnorm._BatchNorm)]
    assert bn_layers and all(item.training is False for item in bn_layers)


def test_v3_2_phase_has_explicit_teacher_bank_build():
    config = {"dual_shift": {"warm_clean_epochs": 2, "warm_apis_epochs": 2, "alpha_max": 0.25}}
    phases = [phase_schedule(epoch, config, variant="apic_v3_2_x") for epoch in range(6)]
    assert phases[2]["phase"] == "style_bank_build"
    assert phases[2]["prepare_style_bank"] is True
    assert phases[3]["phase"] == "apis_warmup"
    assert phases[5]["phase"] == "joint"
