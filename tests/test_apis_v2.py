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


# The current branch lacks several legacy modules imported by Model/__init__.py.
# Load the independently testable APIS components without that package entry.
apis = _load_module("apis_under_test", "Model/dual_shift/apis.py")
protocols = _load_module(
    "protocol_prototypes_under_test", "Model/dual_shift/protocol_prototypes.py"
)
APISModule = apis.APISModule
ProtocolResidualOperator = apis.ProtocolResidualOperator
ProtocolPrototypeBank = protocols.ProtocolPrototypeBank


class ProtocolResidualOperatorTest(unittest.TestCase):
    def test_zero_condition_is_identity(self):
        operator = ProtocolResidualOperator(8, 12, basis_count=3, rank=4)
        features = torch.randn(2, 8, 4, 4, 4)
        shifted, audit = operator(features, torch.zeros(2, 12), strength=0.25)
        torch.testing.assert_close(shifted, features)
        self.assertEqual(float(audit["coefficient_l2"]), 0.0)

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
        (shifted.mean() + audit["coefficient_l2"]).backward()
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(condition.grad)
        self.assertIsNotNone(operator.controller.weight.grad)


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


class ProtocolBankTest(unittest.TestCase):
    def test_samples_only_an_observed_different_protocol(self):
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
        target, selected = bank.sample_target_embeddings(
            ["Siemens|1.5T|A|MPRAGE"], device=torch.device("cpu")
        )
        self.assertIsNotNone(target)
        torch.testing.assert_close(target, torch.ones(1, 4))
        self.assertGreaterEqual(int(selected.item()), 0)
        self.assertEqual(
            bank.last_selection_audit[0]["intervention"], "bounded_residual"
        )


if __name__ == "__main__":
    unittest.main()
