import pytest
import torch
import torch.nn as nn
from pathlib import Path

from Code.models.deep_learning.hrm import (
    HierarchicalReasoningModel,
    HRMClassifierWrapper,
    HRMConfig,
    HRMPipelineBuilder,
)
from Code.models.models import Pipeline


@pytest.fixture
def basic_hrm_config():
    return HRMConfig(
        batch_size=2,
        seq_len=10,
        vocab_size=100,
        hidden_size=16,
        output_embed_dim=4,
        H_cycles=2,
        L_cycles=2,
        halt_max_steps=5,
        h_level_model="EncoderLM",
        l_level_model="EncoderLM",
        model_kwargs={"num_layers": 1, "num_heads": 2},
    )


def test_hrm_initialization(basic_hrm_config):
    model = HierarchicalReasoningModel(config=basic_hrm_config)
    assert model.H_level is not None
    assert model.L_level is not None
    assert model.output_projection.out_features == 4
    assert model.lm_head.out_features == 100
    assert model.H_init.shape == (16,)
    assert model.L_init.shape == (16,)


def test_hrm_forward_sentence_embedding(basic_hrm_config):
    model = HierarchicalReasoningModel(config=basic_hrm_config)
    model.train()
    x = torch.randn(2, 10, 16)
    emb = model(x, pretrain=False)
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (2, 4)


def test_hrm_forward_mlm(basic_hrm_config):
    model = HierarchicalReasoningModel(config=basic_hrm_config)
    model.train()
    x = torch.randn(2, 10, 16)
    logits = model(x, pretrain=True)
    assert logits.shape == (2, 10, 100)


def test_hrm_classifier_wrapper(basic_hrm_config):
    enc = HierarchicalReasoningModel(config=basic_hrm_config)
    wrapped = HRMClassifierWrapper(enc, n_classes=5)
    wrapped.train()
    x = torch.randn(2, 10, 16)
    logits = wrapped(x, pretrain=False)
    assert logits.shape == (2, 5)
    assert wrapped(x, pretrain=True).shape == (2, 10, 100)


def test_hrm_pipeline_builder(basic_hrm_config):
    template_name = "test_hrm_pipeline"

    pipeline = HRMPipelineBuilder.create_hrm_pipeline(
        template_name=template_name,
        hrm_config=basic_hrm_config,
        n_classes=5,
    )

    assert isinstance(pipeline, Pipeline)
    assert "HRM_CORE" in pipeline.blocks

    expected_path = Path("./configs/models") / f"{template_name}.mmd"
    if not expected_path.exists():
        expected_path = Path("configs/models") / f"{template_name}.mmd"
    assert expected_path.exists(), f"missing {expected_path.resolve()}"

    content = expected_path.read_text(encoding="utf-8")
    assert "graph TD" in content
    assert "HRM_CORE" in content
    assert "H-Cycles" in content

    x = torch.randn(2, 10, 16)
    pipeline.train()
    out = pipeline(x)
    assert out.shape == (2, 5)

    expected_path.unlink()
