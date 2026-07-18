from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.model.base import BaseTransformer, grouped_query_attention, repeat_kv_heads
from src.model.config import ModelConfig, RoutedModelConfig
from src.model.moe import MoETransformer
from src.run.main import get_routed_retain_targets
from src.run.train.routed import UnorderedConfig
from src.run.util.config import (
    DataConfig,
    DataLabelConfig,
    ExperimentConfig,
    RunConfig,
    resolve_device,
    resolve_dtype,
    use_fused_adamw,
)
from src.run.util.dataloader import SingleDataLoader


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_resolution_order(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("auto").type, "cuda")
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto").type, "mps")
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto").type, "cpu")

    def test_unavailable_explicit_devices_fail(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                resolve_device("cuda")
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
                resolve_device("mps")

    def test_dtype_resolution(self):
        self.assertEqual(resolve_dtype("auto", torch.device("cuda")), torch.bfloat16)
        self.assertEqual(resolve_dtype("auto", torch.device("mps")), torch.float32)
        self.assertEqual(resolve_dtype("auto", torch.device("cpu")), torch.float32)
        self.assertEqual(resolve_dtype("fp32", torch.device("cpu")), torch.float32)

    def test_optimizer_fusion_is_cuda_only(self):
        self.assertTrue(use_fused_adamw("cuda"))
        self.assertFalse(use_fused_adamw("mps"))
        self.assertFalse(use_fused_adamw("cpu"))

    def test_non_cuda_loader_is_unpinned_and_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample_train.bin"
            np.arange(65, dtype=np.uint16).tofile(path)
            loader = SingleDataLoader(
                str(path), B=2, T=8, device=torch.device("cpu"), pin_memory=False
            )
            batch, _ = loader.next_batch()
            self.assertFalse(loader.dataloader.pin_memory)
            self.assertEqual(batch.device.type, "cpu")


class AttentionTests(unittest.TestCase):
    def _check_attention(self, device: torch.device):
        torch.manual_seed(7)
        q = torch.randn(2, 8, 6, 4, device=device, dtype=torch.float32)
        k = torch.randn(2, 2, 6, 4, device=device, dtype=torch.float32)
        v = torch.randn(2, 2, 6, 4, device=device, dtype=torch.float32)
        mask = torch.ones(2, 1, 6, 6, device=device, dtype=torch.bool).tril()
        actual = grouped_query_attention(q, k, v, mask)
        repeated_k = repeat_kv_heads(k, 8)
        repeated_v = repeat_kv_heads(v, 8)
        scores = q @ repeated_k.transpose(-2, -1) / (q.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ repeated_v
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_repeated_kv_matches_explicit_attention_cpu(self):
        self._check_attention(torch.device("cpu"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_repeated_kv_matches_explicit_attention_mps(self):
        self._check_attention(torch.device("mps"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_native_cuda_gqa_matches_repeated_kv(self):
        q = torch.randn(2, 8, 6, 4, device="cuda")
        k = torch.randn(2, 2, 6, 4, device="cuda")
        v = torch.randn(2, 2, 6, 4, device="cuda")
        native = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        repeated = torch.nn.functional.scaled_dot_product_attention(
            q, repeat_kv_heads(k, 8), repeat_kv_heads(v, 8)
        )
        torch.testing.assert_close(native, repeated, atol=1e-5, rtol=1e-5)


class ModelStepTests(unittest.TestCase):
    def setUp(self):
        self.base_config = ModelConfig(
            ctx_len=8, vocab_size=32, num_layers=1, embed_dim=32,
            mlp_dim=128, num_heads=4, num_key_value=2,
            attn_bias=True, eos_token_id=1,
        )

    def _dense_step(self, device: torch.device):
        model = BaseTransformer(self.base_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=device.type == "cuda")
        tokens = torch.randint(0, 32, (2, 8), device=device)
        _, loss = model(tokens, tokens)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def _gram_step(self, device: torch.device):
        config = RoutedModelConfig.from_base(
            self.base_config, arch="moe", core_param_prc=1.0, aux_param_prc=0.5
        )
        labels = ["core", "aux"]
        model = MoETransformer(config, labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=device.type == "cuda")
        tokens = torch.randint(0, 32, (2, 8), device=device)
        mask = torch.ones(2, dtype=torch.bool, device=device)
        _, loss = model(tokens, tokens, fwd_mask=mask, bck_mask=mask)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_dense_and_gram_steps_cpu(self):
        self._dense_step(torch.device("cpu"))
        self._gram_step(torch.device("cpu"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_dense_and_gram_steps_mps(self):
        self._dense_step(torch.device("mps"))
        self._gram_step(torch.device("mps"))


class RoutedProfilesTests(unittest.TestCase):
    def test_explicit_retain_targets_override(self):
        profiles = [["core", "a", "b"], ["core", "b"]]
        stage = UnorderedConfig(
            model=RoutedModelConfig(arch="moe"), retain_targets=profiles
        )
        config = ExperimentConfig(
            data=DataConfig(aux=DataLabelConfig(labels=["a", "b"])),
            run=RunConfig(labels=["core", "a", "b"]),
        )
        self.assertIs(get_routed_retain_targets(stage, config), profiles)


if __name__ == "__main__":
    unittest.main()
