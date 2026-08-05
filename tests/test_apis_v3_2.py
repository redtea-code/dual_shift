import torch
import pytest

from Model.dual_shift.apis_v3_2 import FixedStyleBankAPICV32
from Model.dual_shift.backbone import DualShiftBackbone
from Model.dual_shift.losses import compute_dual_shift_loss
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


def test_v3_2_module_to_device():
    """Regression: do not shadow nn.Module._apply (breaks .to/.cuda)."""
    module = _module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = module.to(device)
    buf = next(moved.buffers())
    assert buf.device.type == device.type
    assert "prototype_pair_relative_distance" not in moved.state_dict()


def test_v3_2_relative_pair_distance_is_scale_invariant_and_enables_support():
    module = _module()
    module._finalized = True
    module.style_valid[:] = True
    module.style_prototypes[:] = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    )
    module.prototype_radii[:] = torch.tensor([10.0, 10.0])
    source = torch.tensor([0])
    target = torch.tensor([1])
    assert torch.allclose(module._relative_pair_distance(source, target), torch.tensor([2.0]))
    module.style_prototypes.mul_(5.0)
    module.prototype_radii.mul_(5.0)
    assert torch.allclose(module._relative_pair_distance(source, target), torch.tensor([2.0]))


def test_v3_2_relative_delta_band_selects_a_real_alternative():
    module = _module()
    module._finalized = True
    module.style_valid[:] = True
    module.style_prototypes[:] = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    )
    module.prototype_radii[:] = torch.tensor([10.0, 10.0])
    module.descriptor_center.zero_()
    module.descriptor_scale.fill_(1.0)
    module.pca_mean.zero_()
    module.pca_components.zero_()
    module.pca_components[0, 0] = 1.0
    image = torch.randn(2, 1, 8, 8, 8)
    layer1 = torch.zeros(2, 2, 4, 4, 4)
    layer2 = torch.zeros(2, 4, 2, 2, 2)
    _, valid = module.prepare_style_condition(image, layer1, layer2, sample_ids=["a", "b"])
    assert valid.all()
    assert torch.equal(module._state["src"], torch.zeros(2, dtype=torch.long))
    assert torch.equal(module._state["target"], torch.ones(2, dtype=torch.long))


def test_v3_2_strict_bank_rejects_missing_calibration():
    module = _module()
    layer1 = torch.randn(8, 2, 4, 4, 4)
    layer2 = torch.randn(8, 4, 2, 2, 2)
    # Select subjects deterministically assigned to mechanism_fit only.
    ids = ["subject-0", "subject-1", "subject-2", "subject-3"]
    while any(
        int(__import__("hashlib").sha256(item.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0
        for item in ids
    ):
        ids = [item + "x" for item in ids]
    module.observe_source(layer1[:4], layer2[:4], subject_ids=ids)
    with pytest.raises(RuntimeError, match="mechanism_calibration is empty"):
        module.finalize_style_bank(strict=True)


def test_v3_2_loss_counts_every_sample_once_and_auxiliary_terms_use_full_batch():
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    shifted_logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    mask = torch.tensor([1.0, 0.0])
    clean_embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    shifted_embedding = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    losses = compute_dual_shift_loss(
        clean_logits=clean_logits,
        shifted_logits=shifted_logits,
        labels=labels,
        intervention_mask=mask,
        clean_embedding=clean_embedding,
        shifted_embedding=shifted_embedding,
        lambda_js=0.0,
        lambda_feat=1.0,
        v3_2_mode=True,
    )
    clean_per = torch.nn.functional.cross_entropy(clean_logits, labels, reduction="none")
    shifted_per = torch.nn.functional.cross_entropy(shifted_logits, labels, reduction="none")
    expected_cls = (0.5 * (clean_per[0] + shifted_per[0]) + clean_per[1]) / 2.0
    # Supported cosine distance is 1, but revision-4 averages auxiliaries over
    # the whole batch, hence 0.5 rather than 1.
    assert torch.allclose(losses["shift_ce"], expected_cls)
    assert torch.allclose(losses["feat"], torch.tensor(0.5))
    assert torch.allclose(losses["total"], expected_cls + torch.tensor(0.5))


def test_v3_2_zero_support_is_exact_clean_ce():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    losses = compute_dual_shift_loss(
        clean_logits=logits,
        shifted_logits=logits.flip(1),
        labels=labels,
        intervention_mask=torch.zeros(2),
        v3_2_mode=True,
    )
    assert torch.allclose(losses["total"], losses["clean_ce"])
