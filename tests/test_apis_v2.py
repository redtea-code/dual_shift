import importlib.util
import sys
import unittest
from pathlib import Path

import torch


def _load_module(name, relative_path):
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, repo_root / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load independently testable APIS components without requiring every legacy
# Model package export to succeed.
apis = _load_module("apis_under_test", "Model/dual_shift/apis.py")
protocols = _load_module(
    "protocol_prototypes_under_test", "Model/dual_shift/protocol_prototypes.py"
)
losses = _load_module("losses_under_test", "Model/dual_shift/losses.py")
APISModule = apis.APISModule
ProtocolResidualOperator = apis.ProtocolResidualOperator
ProtocolPrototypeBank = protocols.ProtocolPrototypeBank
compute_dual_shift_loss = losses.compute_dual_shift_loss


class ProtocolResidualOperatorTest(unittest.TestCase):
    def test_zero_condition_is_identity(self):
        operator = ProtocolResidualOperator(8, 12, basis_count=3, rank=4)
        features = torch.randn(2, 8, 4, 4, 4)
        shifted, audit = operator(features, torch.zeros(2, 12), strength=0.25)
        torch.testing.assert_close(shifted, features)
        self.assertTrue(torch.allclose(audit["coefficient_l2"], torch.zeros(2)))

    def test_residual_is_bounded_and_differentiable(self):
        operator = ProtocolResidualOperator(8, 12, basis_count=3, rank=4)
        features = torch.randn(2, 8, 4, 4, 4, requires_grad=True)
        condition = torch.randn(2, 12, requires_grad=True)
        shifted, audit = operator(features, condition, strength=0.2)
        relative_rms = (
            (shifted - features).flatten(1).square().mean(1).sqrt()
            / features.flatten(1).square().mean(1).sqrt().clamp_min(1e-6)
        )
        self.assertTrue(torch.all(relative_rms <= 0.20001))
        (shifted.mean() + audit["coefficient_l2"].mean()).backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(condition.grad)
        self.assertIsNotNone(operator.controller.weight.grad)

    def test_valid_mask_keeps_invalid_rows_unshifted(self):
        module = APISModule(
            layer1_channels=8,
            layer2_channels=16,
            acquisition_dim=4,
            basis_count=2,
            rank=4,
        )
        module.set_alpha(0.2)
        features = torch.randn(3, 8, 4, 4, 4)
        condition = torch.randn(3, 12)
        mask = torch.tensor([True, False, True])
        shift1, _ = module.make_shift_fns(condition, valid_mask=mask)
        shifted = shift1(features)
        torch.testing.assert_close(shifted[1], features[1])
        self.assertFalse(torch.allclose(shifted[0], features[0]))


class APISModuleTest(unittest.TestCase):
    def test_condition_is_directed(self):
        module = APISModule(
            layer1_channels=8,
            layer2_channels=16,
            acquisition_dim=4,
            basis_count=2,
            rank=4,
        )
        factual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        condition = module.protocol_condition(factual, target)
        self.assertEqual(tuple(condition.shape), (1, 12))
        torch.testing.assert_close(condition[:, -4:], target - factual)


class BoundedAcquisitionFiLMTest(unittest.TestCase):
    def test_starts_as_identity_and_is_bounded(self):
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from Model.dual_shift.model import BoundedAcquisitionFiLM  # noqa: WPS433

        module = BoundedAcquisitionFiLM(4, 8, alpha=0.1)
        features = torch.randn(3, 8, 3, 3, 3)
        embedding = torch.randn(3, 4)
        identity, strength = module(features, embedding)
        torch.testing.assert_close(identity, features)
        self.assertTrue(torch.allclose(strength, torch.zeros_like(strength)))

        with torch.no_grad():
            module.controller.bias.fill_(8.0)
        shifted, strength = module(features, embedding)
        self.assertTrue(torch.all(strength <= 0.21))
        self.assertFalse(torch.allclose(shifted, features))


class ProtocolBankTest(unittest.TestCase):
    def _bank(self):
        bank = ProtocolPrototypeBank(min_subjects=1)
        template = {
            "mean1": torch.zeros(2),
            "std1": torch.ones(2),
            "mean2": torch.zeros(4),
            "std2": torch.ones(4),
            "n_updates": 1,
        }
        bank.prototypes = {
            "Siemens|1.5T|A|MPRAGE": {
                **template,
                "embedding": torch.zeros(4),
                "subjects": {"s1"},
            },
            "Siemens|3T|B|MPRAGE": {
                **template,
                "embedding": torch.ones(4),
                "subjects": {"s2"},
            },
        }
        return bank

    def test_samples_only_an_observed_different_protocol(self):
        bank = self._bank()
        target, selected = bank.sample_target_embeddings(
            ["Siemens|1.5T|A|MPRAGE"], device=torch.device("cpu")
        )
        self.assertIsNotNone(target)
        torch.testing.assert_close(target, torch.ones(1, 4))
        self.assertGreaterEqual(int(selected.item()), 0)
        self.assertEqual(
            bank.last_selection_audit[0]["intervention"], "bounded_residual"
        )

    def test_keeps_partial_batch_with_per_sample_invalid_mask(self):
        bank = ProtocolPrototypeBank(min_subjects=1)
        bank.prototypes = {
            "Siemens|1.5T|A|MPRAGE": {
                "mean1": torch.zeros(2),
                "std1": torch.ones(2),
                "mean2": torch.zeros(4),
                "std2": torch.ones(4),
                "n_updates": 1,
                "embedding": torch.ones(4),
                "subjects": {"s1"},
            }
        }
        # First key can jump to the only observed protocol; the second key is that
        # protocol itself and therefore has no valid residual target.
        target, selected = bank.sample_target_embeddings(
            ["GE|3T|Z|OTHER", "Siemens|1.5T|A|MPRAGE"],
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(target)
        self.assertEqual(tuple(target.shape), (2, 4))
        self.assertGreaterEqual(int(selected[0].item()), 0)
        self.assertEqual(int(selected[1].item()), -1)
        self.assertEqual(bank.last_selection_audit[1]["intervention"], "none")
        self.assertEqual(
            bank.last_selection_audit[1]["reason"], "no_valid_observed_protocol"
        )


class InterventionLossTest(unittest.TestCase):
    def test_coefficient_penalty_enters_total_loss(self):
        logits = torch.randn(4, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1])
        shifted = logits + 0.1
        penalty = torch.tensor(2.0, requires_grad=True)
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        losses = compute_dual_shift_loss(
            clean_logits=logits,
            labels=labels,
            shifted_logits=shifted,
            enable_apis=True,
            intervention_penalty=penalty,
            lambda_intervention=0.5,
            intervention_mask=mask,
        )
        self.assertAlmostEqual(float(losses["intervention"]), 2.0)
        losses["total"].backward()
        self.assertIsNotNone(penalty.grad)
        self.assertAlmostEqual(float(penalty.grad), 0.5)

    def test_all_invalid_mask_disables_shift_terms(self):
        logits = torch.randn(3, 2)
        labels = torch.tensor([0, 1, 0])
        shifted = logits + 1.0
        losses = compute_dual_shift_loss(
            clean_logits=logits,
            labels=labels,
            shifted_logits=shifted,
            enable_apis=True,
            intervention_penalty=torch.tensor(3.0),
            intervention_mask=torch.zeros(3),
        )
        self.assertEqual(float(losses["shift_ce"]), 0.0)
        self.assertEqual(float(losses["js"]), 0.0)
        self.assertEqual(float(losses["intervention"]), 0.0)


class PackageImportTest(unittest.TestCase):
    def test_dual_shift_import_survives_model_package(self):
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from Model.dual_shift import DualShiftResNet3D  # noqa: WPS433

        self.assertTrue(callable(DualShiftResNet3D))


if __name__ == "__main__":
    unittest.main()
