"""Tests for HRM embedding smoke tool (config + safetensors + sample forward)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]


def test_strip_module_prefix():
    from Code.thesis.tools.hrm_embed_smoke import _strip_module_prefix

    sd = {"module.w": torch.zeros(1), "bias": torch.ones(2)}
    out = _strip_module_prefix(sd)
    assert set(out.keys()) == {"w", "bias"}


def test_hrm_embed_smoke_cli_exits_when_checkpoint_missing(tmp_path):
    """Repo config exists; point checkpoint at missing file → exit code 2."""
    cfg = _REPO / "Code" / "thesis" / "config" / "hrm" / "E_HRM1_4Level.json"
    if not cfg.is_file():
        pytest.skip("HRM config not in tree")
    missing_ckpt = tmp_path / "nope.safetensors"
    script = _REPO / "Code" / "thesis" / "tools" / "hrm_embed_smoke.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(cfg),
            "--checkpoint",
            str(missing_ckpt),
            "--device",
            "cpu",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_hrm_embed_smoke_tiny_roundtrip(tmp_path):
    """Small HRM: save safetensors, reload via load_hrm_encoder, tensor embedding forward."""
    from Code.thesis.tools.hrm_embed_smoke import ensure_hrm_encoder_imports_primed

    ensure_hrm_encoder_imports_primed()
    from Code.models.deep_learning.hrm.hrm_model import HierarchicalReasoningModel, HRMConfig
    from Code.thesis.common.checkpoint_io import save_safetensors
    from Code.thesis.tools.hrm_embed_smoke import load_hrm_encoder

    hcfg = HRMConfig(
        batch_size=1,
        seq_len=8,
        vocab_size=64,
        hidden_size=32,
        output_embed_dim=8,
        H_cycles=1,
        L_cycles=1,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        model_kwargs={"num_layers": 1, "num_heads": 4},
        tokenizer_name=None,
    )
    m0 = HierarchicalReasoningModel(config=hcfg)
    ckpt = tmp_path / "enc.safetensors"
    save_safetensors(m0.state_dict(), ckpt)

    cfg_dict = {
        "HierarchicalReasoningModel": {
            "config": {
                "batch_size": 1,
                "seq_len": 8,
                "hidden_size": 32,
                "output_embed_dim": 8,
                "vocab_size": 64,
                "H_cycles": 1,
                "L_cycles": 1,
                "h_level_model": "EncoderLM",
                "l_level_model": "EncoderLM",
                "model_kwargs": {"num_layers": 1, "num_heads": 4},
            }
        }
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_dict), encoding="utf-8")

    model = load_hrm_encoder(cfg_path, ckpt, torch.device("cpu"), tokenizer_dir=None)
    x = torch.randn(2, 8, 32)
    model.eval()
    with torch.no_grad():
        emb = model(x, pretrain=False)
    assert emb.shape == (2, 8)


def test_hrm_embed_texts_tiny_cpu():
    """embed_texts() on list[str] with tokenizer (downloads BERT once if needed)."""
    pytest.importorskip("transformers")
    from Code.thesis.tools.hrm_embed_smoke import ensure_hrm_encoder_imports_primed

    ensure_hrm_encoder_imports_primed()
    from Code.models.deep_learning.hrm.hrm_model import HierarchicalReasoningModel, HRMConfig
    from Code.thesis.tools.hrm_embed_smoke import embed_texts

    hcfg = HRMConfig(
        batch_size=1,
        seq_len=32,
        vocab_size=30522,
        hidden_size=32,
        output_embed_dim=8,
        H_cycles=1,
        L_cycles=1,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        tokenizer_name="google-bert/bert-base-uncased",
        model_kwargs={"num_layers": 1, "num_heads": 4},
    )
    model = HierarchicalReasoningModel(config=hcfg)
    model.eval()
    texts = ["hello world", "another phrase"]
    emb = embed_texts(model, texts)
    assert emb.shape == (2, 8)
