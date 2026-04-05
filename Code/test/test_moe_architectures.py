import pytest
import torch
import torch.nn as nn
from Code.models.moe.parallel import ParallelMoE
from Code.models.moe.sequential import SequentialMoE
from Code.models.moe.experts import ExpertWrapper
from Code.models.moe.builder import MoEPipelineBuilder
from Code.models.models import Pipeline
from pathlib import Path


class DummyHRM(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.net(x)


class DummyExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)
    def forward(self, x):
        return self.net(x)


class DummyMLExpert:
    def predict(self, X):
        # returns simple dummy prediction shape (batch_size, 1) or similar
        return X.sum(axis=-1, keepdims=True)


def test_expert_wrapper():
    ml_model = DummyMLExpert()
    wrapper = ExpertWrapper(ml_model, out_features=1)
    
    x = torch.randn(4, 10)
    out = wrapper(x)
    
    assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"
    assert out.requires_grad == False # ML models shouldn't track gradients
    

def test_parallel_moe_forward():
    in_features = 10
    out_features = 2
    batch_size = 4
    
    experts = {
        "dl_expert_1": DummyExpert(in_features, out_features),
        "dl_expert_2": DummyExpert(in_features, out_features),
    }
    
    moe = ParallelMoE(
        experts=nn.ModuleDict(experts),
        in_features=in_features,
        hrm=None
    )
    
    x = torch.randn(batch_size, in_features)
    out = moe(x)
    
    assert out.shape == (batch_size, out_features), f"Expected shape (4, 2), got {out.shape}"
    assert out.requires_grad == True # Differentiable


def test_parallel_moe_with_hrm():
    in_features = 10
    hrm_features = 5
    out_features = 2
    batch_size = 4
    
    hrm = DummyHRM(in_features, hrm_features)
    
    experts = {
        "dl_1": DummyExpert(in_features, out_features),
        "dl_2": DummyExpert(in_features, out_features),
    }
    
    moe = ParallelMoE(
        experts=nn.ModuleDict(experts),
        in_features=hrm_features,
        hrm=hrm
    )
    
    x = torch.randn(batch_size, in_features)
    out = moe(x)
    
    assert out.shape == (batch_size, out_features)


def test_sequential_moe_forward():
    in_features = 10
    out_features = 2
    batch_size = 4
    
    # We set weights heavily so they produce predictable softmax distributions
    expert1 = DummyExpert(in_features, out_features)
    expert2 = DummyExpert(in_features, out_features)
    
    moe = SequentialMoE(
        experts=nn.ModuleList([expert1, expert2]),
        thresholds=0.5 # 50% confidence
    )

    x = torch.randn(batch_size, in_features)
    out = moe(x)
    
    assert out.shape == (batch_size, out_features)


def test_moe_builder_parallel():
    in_features = 10
    out_features = 2
    
    experts = {
        "dl_expert_1": DummyExpert(in_features, out_features),
        "dl_expert_2": DummyExpert(in_features, out_features),
    }
    
    # Create the pipeline using builder
    pipeline = MoEPipelineBuilder.create_parallel_moe_pipeline(
        name="test_parallel_pipeline",
        experts=experts,
        in_features=in_features
    )
    
    assert isinstance(pipeline, Pipeline)
    
    x = torch.randn(4, in_features)
    out = pipeline(x)
    
    assert out.shape == (4, out_features)


def test_moe_builder_sequential():
    in_features = 10
    out_features = 2
    
    experts = [
        DummyExpert(in_features, out_features),
        DummyExpert(in_features, out_features),
    ]
    
    # Create the pipeline using builder
    pipeline = MoEPipelineBuilder.create_sequential_moe_pipeline(
        name="test_sequential_pipeline",
        experts=experts,
        thresholds=0.8
    )
    
    assert isinstance(pipeline, Pipeline)
    
    x = torch.randn(4, in_features)
    out = pipeline(x)
    
    assert out.shape == (4, out_features)
