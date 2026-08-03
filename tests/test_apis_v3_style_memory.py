import unittest

import torch

from Model.backbone.journal_resnet import journal_resnet10
from Model.dual_shift.apis_v3 import StyleMemoryAPISV3
from Model.dual_shift.model import DualShiftResNet3D


class StyleMemoryAPISV3Test(unittest.TestCase):
    def test_image_only_memory_is_bounded_and_differentiable(self):
        torch.manual_seed(11)
        module = StyleMemoryAPISV3(
            layer1_channels=4,
            layer2_channels=8,
            layer3_channels=16,
            alpha_max=0.25,
            memory_size=4,
        )
        module.train()
        module.set_alpha(0.2)
        for _ in range(2):
            layer1 = torch.randn(6, 4, 4, 4, 4, requires_grad=True)
            layer2 = torch.randn(6, 8, 2, 2, 2, requires_grad=True)
            layer3 = torch.randn(6, 16, 1, 1, 1, requires_grad=True)
            condition, valid = module.prepare_style_condition(
                layer1, layer2, layer3, update_memory=True
            )

        shift1, shift2 = module.make_shift_fns(condition, valid_mask=valid)
        shifted1 = shift1(layer1)
        shifted2 = shift2(layer2)
        audit = module.audit_tensors(layer1)

        self.assertEqual(condition.shape, (6, module.condition_dim))
        self.assertGreaterEqual(int(module.style_valid.sum()), 2)
        self.assertTrue(bool(valid.any()))
        self.assertLessEqual(float(audit["strength"]), 0.2 + 1e-6)
        (shifted1.mean() + shifted2.mean() + condition.mean()).backward()
        self.assertIsNotNone(module.style_encoder[1].weight.grad)
        self.assertIsNotNone(module.layer1_operator.up.weight.grad)

    def test_model_runs_without_acquisition_metadata(self):
        torch.manual_seed(13)
        model = DualShiftResNet3D(
            num_classes=2,
            layers=(1, 1, 1, 1),
            base_channels=4,
            acquisition_out_dim=4,
            apis_variant="v3_style_memory",
        )
        model.train()
        model.set_phase(apis_active=True, cdt_enabled=False, alpha=0.2)
        image = torch.randn(4, 1, 16, 16, 16)
        covariates = torch.zeros(4, 4)

        output = None
        for _ in range(3):
            output = model(image, covariates=covariates, acquisitions=None)

        self.assertIsNotNone(output.shifted_logits)
        self.assertEqual(tuple(output.shifted_logits.shape), (4, 2))
        self.assertEqual(output.extras["apis_mode"], "v3_style_memory")
        self.assertIn("style_entropy", output.extras)

        model.eval()
        clean = model(image, covariates=covariates, acquisitions=None)
        self.assertIsNone(clean.shifted_logits)

    def test_image_only_mode_ignores_demographics(self):
        torch.manual_seed(17)
        model = DualShiftResNet3D(
            num_classes=2,
            layers=(1, 1, 1, 1),
            base_channels=4,
            acquisition_out_dim=4,
            use_demographics=False,
            apis_variant="v3_style_memory",
        )
        model.eval()
        image = torch.randn(2, 1, 16, 16, 16)
        first = model(image, covariates=torch.zeros(2, 3), acquisitions=None)
        second = model(image, covariates=torch.full((2, 3), 99.0), acquisitions=None)

        self.assertTrue(torch.equal(first.clean_logits, second.clean_logits))

    def test_image_only_journal_baseline_ignores_demographics(self):
        torch.manual_seed(19)
        model = journal_resnet10(
            num_classes=2,
            base_channels=4,
            spatial_shape=(1, 1, 1),
            var_specs=[],
        ).eval()
        image = torch.randn(2, 1, 16, 16, 16)
        first = model(image, torch.zeros(2, 3))
        second = model(image, torch.full((2, 3), -5.0))

        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
