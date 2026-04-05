"""HRM encoder-only factory + MLM one-step backward (CPU)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from Code.models.deep_learning.hrm.hrm_model import (
    HRMClassifierWrapper,
    HRMConfig,
    HierarchicalReasoningModel,
)
from Code.thesis.common.model_factory import build_model_from_config_dict


def test_build_hrm_encoder_only_no_wrapper() -> None:
    cfg = {
        "HierarchicalReasoningModel": {
            "config": {
                "batch_size": 2,
                "seq_len": 8,
                "hidden_size": 32,
                "output_embed_dim": 16,
                "vocab_size": 128,
                "H_cycles": 1,
                "L_cycles": 1,
                "h_level_model": "EncoderLM",
                "l_level_model": "EncoderLM",
                "tokenizer_name": "google-bert/bert-base-uncased",
                "model_kwargs": {"num_layers": 1, "num_heads": 4},
            }
        }
    }
    enc, name = build_model_from_config_dict(cfg, 2, "", hrm_encoder_only=True)
    assert name == "HierarchicalReasoningModel"
    assert isinstance(enc, HierarchicalReasoningModel)
    assert not isinstance(enc, HRMClassifierWrapper)

    wrapped, _ = build_model_from_config_dict(cfg, 2, "", hrm_encoder_only=False)
    assert isinstance(wrapped, HRMClassifierWrapper)


def test_hrm_mlm_one_step_backward_cpu() -> None:
    hcfg = HRMConfig(
        batch_size=1,
        seq_len=8,
        vocab_size=64,
        hidden_size=32,
        output_embed_dim=8,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=3,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        model_kwargs={"num_layers": 1, "num_heads": 4},
    )
    model = HierarchicalReasoningModel(config=hcfg)
    model.train()
    x = torch.randn(2, 8, 32)
    logits = model(x, pretrain=True)
    assert logits.shape == (2, 8, 64)
    loss = logits.float().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hrm_mlm_one_step_backward_cuda() -> None:
    hcfg = HRMConfig(
        batch_size=1,
        seq_len=8,
        vocab_size=64,
        hidden_size=32,
        output_embed_dim=8,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=3,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        model_kwargs={"num_layers": 1, "num_heads": 4},
    )
    device = torch.device("cuda:0")
    model = HierarchicalReasoningModel(config=hcfg).to(device)
    model.train()
    x = torch.randn(2, 8, 32, device=device)
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(x, pretrain=True)
        loss = logits.float().mean()
    loss.backward()
    torch.cuda.synchronize()


def test_hrm_pretrain_safetensors_roundtrip(tmp_path: Path) -> None:
    from Code.thesis.common.checkpoint_io import load_safetensors_state, save_safetensors

    hcfg = HRMConfig(
        batch_size=1,
        seq_len=4,
        vocab_size=32,
        hidden_size=16,
        output_embed_dim=8,
        H_cycles=1,
        L_cycles=1,
        halt_max_steps=2,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        model_kwargs={"num_layers": 1, "num_heads": 4},
    )
    m1 = HierarchicalReasoningModel(config=hcfg)
    path = tmp_path / "enc.safetensors"
    save_safetensors(m1.state_dict(), path)
    m2 = HierarchicalReasoningModel(config=hcfg)
    sd = load_safetensors_state(str(path), map_location="cpu")
    m2.load_state_dict(sd, strict=True)
    x = torch.randn(1, 4, 16)
    torch.testing.assert_close(m1(x, pretrain=True), m2(x, pretrain=True))
