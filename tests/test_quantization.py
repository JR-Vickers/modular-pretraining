from __future__ import annotations

import unittest

import torch

from analysis.stories_phase2.common import DATA_LABELS
from src.model.base import BaseTransformer
from src.model.config import ModelConfig, RoutedModelConfig
from src.model.moe import MoETransformer
from src.run.quantization import GROUPS, classify_matrix_weights, fake_quantize_tensor, quantize_model_copy


def base_config() -> ModelConfig:
    return ModelConfig(ctx_len=8, vocab_size=16, num_layers=1, num_heads=2,
                       num_key_value=1, embed_dim=8, mlp_dim=64, eos_token_id=0)


def gram_config() -> RoutedModelConfig:
    return RoutedModelConfig.from_base(base_config(), core_param_prc=1.0, aux_param_prc=1.0)


class QuantizationTests(unittest.TestCase):
    def test_known_tensor_uses_narrow_symmetric_grid(self) -> None:
        weight = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0], [1.5, 0.5, 0.0, -0.5, -1.5]])
        actual, scale = fake_quantize_tensor(weight, 3, "per_channel")
        expected = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0], [1.5, 0.5, 0.0, -0.5, -1.5]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(scale, torch.tensor([[1.0], [0.5]]))

    def test_zero_rows_and_tensor_are_exact(self) -> None:
        weight = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        actual, scale = fake_quantize_tensor(weight, 4, "per_channel")
        self.assertTrue(torch.equal(actual[0], weight[0]))
        self.assertEqual(scale.shape, (2, 1))
        zero, tensor_scale = fake_quantize_tensor(torch.zeros(2, 3), 4, "per_tensor")
        self.assertTrue(torch.equal(zero, torch.zeros(2, 3)))
        self.assertEqual(tensor_scale.shape, (1,))

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            fake_quantize_tensor(torch.ones(2, 2), 1)
        with self.assertRaises(ValueError):
            fake_quantize_tensor(torch.ones(2, 2), 4, "blocks")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fake_quantize_tensor(torch.ones(2), 4)

    def test_per_channel_differs_from_per_tensor(self) -> None:
        weight = torch.tensor([[0.1, 0.0], [10.0, 1.0]])
        per_channel, _ = fake_quantize_tensor(weight, 4, "per_channel")
        per_tensor, _ = fake_quantize_tensor(weight, 4, "per_tensor")
        self.assertFalse(torch.equal(per_channel, per_tensor))

    def test_classification_is_exhaustive_and_non_overlapping(self) -> None:
        gram = classify_matrix_weights(MoETransformer(gram_config(), list(DATA_LABELS)))
        self.assertEqual(set(gram.values()), set(GROUPS))
        dense = classify_matrix_weights(BaseTransformer(base_config()))
        self.assertEqual(set(dense.values()), {"core_mlp", "attention", "embeddings"})
        self.assertEqual(len(gram), len(set(gram)))

    def test_copy_preserves_source_and_unselected_parameters(self) -> None:
        model = MoETransformer(gram_config(), list(DATA_LABELS))
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        quantized, stats = quantize_model_copy(model, 4, selected_groups=["aux_modules"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        assignments = classify_matrix_weights(model)
        for name, value in quantized.state_dict().items():
            if name not in assignments or assignments[name] != "aux_modules":
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertEqual(set(stats["per_group"]), {"aux_modules"})

    def test_inactive_aux_quantization_and_mask_order(self) -> None:
        torch.manual_seed(7)
        model = MoETransformer(gram_config(), list(DATA_LABELS)).eval()
        tokens = torch.tensor([[1, 2, 3, 0]])
        all_on = torch.ones(len(DATA_LABELS), dtype=torch.bool)
        deadline_off = all_on.clone()
        deadline_off[1] = False
        quantized, _ = quantize_model_copy(model, 2, selected_groups=["aux_modules"])
        original_off = model(tokens, None, deadline_off, deadline_off)[0]
        quantized_off = quantized(tokens, None, deadline_off, deadline_off)[0]
        # All aux modules were changed, so isolate the inactive one for exactness.
        deadline_only = MoETransformer(gram_config(), list(DATA_LABELS)).eval()
        deadline_only.load_state_dict(model.state_dict())
        assignments = classify_matrix_weights(model)
        qstate = quantized.state_dict()
        with torch.no_grad():
            for name, parameter in deadline_only.named_parameters():
                if ".moe.experts.1." in name and assignments.get(name) == "aux_modules":
                    parameter.copy_(qstate[name])
        isolated_off = deadline_only(tokens, None, deadline_off, deadline_off)[0]
        torch.testing.assert_close(isolated_off, original_off, rtol=0, atol=0)
        self.assertFalse(torch.equal(quantized(tokens, None, all_on, all_on)[0],
                                     model(tokens, None, all_on, all_on)[0]))
        # Masks are external inputs and quantization does not mutate them.
        self.assertTrue(torch.equal(deadline_off, torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)))
        self.assertEqual(quantized_off.shape, original_off.shape)


if __name__ == "__main__":
    unittest.main()
