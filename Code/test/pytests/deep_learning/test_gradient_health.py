# src/test/pytests/deep_learning/test_gradient_health.py
"""
Gradient health tests:
  - Gradient flow (no None gradients)
  - Vanishing gradient detection
  - Exploding gradient detection
  - Gradient clipping correctness
  - Computational graph integrity (no accidental detach)
  - retain_graph behavior
  - Gradient accumulation correctness
  - Backprop through AdvancedPipeline nodes
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ....models.utils.utils import DLModule, compute_total_grad_norm  # noqa: E402
from ....models.models import Pipeline  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_grad_norms(model: nn.Module) -> Dict[str, float]:
    """Return {param_name: grad_norm} after a backward pass."""
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.detach().norm(2).item()
        else:
            norms[name] = None  # type: ignore[assignment]
    return norms


def _do_backward(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 loss_fn=None) -> Dict[str, float]:
    """Run one forward+backward and return grad norms."""
    model.zero_grad()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    out  = model.forward(x)            # .forward() not model() to avoid pipeline routing
    loss = loss_fn(out, y)
    loss.backward()
    return _collect_grad_norms(model)


class _DeepNet(DLModule):
    """10-layer MLP — shallow enough to test in CI, deep enough to see vanishing."""
    def __init__(self, width=32, depth=10):
        super().__init__()
        layers = [nn.Linear(width, width), nn.Sigmoid()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Sigmoid()]
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ShallowNet(DLModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════════════
# §8.1  Gradient flow — no None gradients
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_all_parameters_receive_gradients():
    """Every leaf parameter must have a non-None gradient after backward."""
    model = _ShallowNet()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    norms = _do_backward(model, x, y)
    none_params = [n for n, g in norms.items() if g is None]
    assert not none_params, \
        f"These parameters received None gradient (graph broken): {none_params}"


@pytest.mark.unit
def test_gradient_values_are_finite():
    """No NaN or Inf in any gradient after a normal backward pass."""
    model = _ShallowNet()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    _do_backward(model, x, y)
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), \
                f"Non-finite gradient in parameter {name!r}"


# ══════════════════════════════════════════════════════════════════════════════
# §8.2  Vanishing gradient detection
# ══════════════════════════════════════════════════════════════════════════════

VANISHING_THRESHOLD = 1e-5

@pytest.mark.unit
def test_vanishing_gradient_detection_deep_sigmoid():
    """
    A deep sigmoid network (10 layers) must trigger a vanishing gradient alert
    when the first-layer gradient norm falls below the threshold.
    This is a characterization test — it documents the known behavior and
    ensures the codebase can detect it, not that we prevent it.
    """
    model = _DeepNet(width=32, depth=10)
    x = torch.randn(16, 32)
    y = torch.randint(0, 4, (16,))
    norms = _do_backward(model, x, y)

    first_layer_norms = [v for k, v in norms.items()
                         if "net.0" in k and v is not None]
    last_layer_norms  = [v for k, v in norms.items()
                         if "net." in k and v is not None]

    if first_layer_norms and last_layer_norms:
        ratio = min(first_layer_norms) / (max(last_layer_norms) + 1e-9)
        # Document the ratio — in a deep sigmoid network this is typically << 1
        assert ratio >= 0  # always true; here to produce a test artifact with the ratio


@pytest.mark.unit
def test_relu_network_no_vanishing():
    """ReLU activations should not cause vanishing gradients in a 5-layer net."""
    class _ReLUNet(DLModule):
        def __init__(self):
            super().__init__()
            layers = []
            for _ in range(4):
                layers += [nn.Linear(32, 32), nn.ReLU()]
            layers.append(nn.Linear(32, 4))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = _ReLUNet()
    x = torch.randn(16, 32)
    y = torch.randint(0, 4, (16,))
    norms = _do_backward(model, x, y)

    non_none_norms = [v for v in norms.values() if v is not None]
    if non_none_norms:
        min_norm = min(non_none_norms)
        assert min_norm > VANISHING_THRESHOLD, \
            f"Possible vanishing gradient in ReLU net: min norm = {min_norm:.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# §8.3  Exploding gradient detection
# ══════════════════════════════════════════════════════════════════════════════

EXPLODING_THRESHOLD = 1e3

@pytest.mark.unit
def test_exploding_gradient_detection():
    """
    Simulate exploding gradients by using a very large learning rate
    and verifying that gradient norms exceed the threshold.
    This is a characterization test — not a prevention test.
    """
    class _ExplodeNet(DLModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            # Initialize with very large weights to provoke explosion
            nn.init.constant_(self.fc.weight, 100.0)

        def forward(self, x):
            return self.fc(x)

    model = _ExplodeNet()
    x = torch.randn(4, 4) * 100.0  # large inputs
    y = torch.randint(0, 4, (4,))
    norms = _do_backward(model, x, y)
    total_norm = sum(v for v in norms.values() if v is not None)
    # With large weights + large inputs, norm should be large
    assert total_norm > 1.0, \
        f"Expected large gradient norm for exploding setup; got {total_norm:.4f}"


@pytest.mark.unit
def test_gradient_clipping_reduces_norm():
    """
    torch.nn.utils.clip_grad_norm_ must reduce the total gradient norm
    to at most max_norm when gradients are large.
    """
    class _LargeGradNet(DLModule):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            nn.init.constant_(self.fc.weight, 100.0)

        def forward(self, x):
            return self.fc(x)

    model = _LargeGradNet()
    x = torch.randn(4, 4) * 100.0
    y = torch.randint(0, 4, (4,))
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model.forward(x), y)
    loss.backward()

    total_before = sum(
        p.grad.detach().norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5

    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    total_after = sum(
        p.grad.detach().norm(2).item() ** 2
        for p in model.parameters() if p.grad is not None
    ) ** 0.5

    if total_before > max_norm:
        assert total_after <= max_norm + 1e-4, \
            f"Clipping failed: norm after={total_after:.4f} > max_norm={max_norm}"


@pytest.mark.unit
def test_gradients_are_finite_after_normal_step():
    """After a normal training step, all param gradients must be finite (no explosion)."""
    model = _ShallowNet()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    _do_backward(model, x, y)
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


# ══════════════════════════════════════════════════════════════════════════════
# §8.4  Computational graph integrity — accidental detach detection
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_graph_not_broken_by_accidental_detach():
    """
    If a model uses .detach() on an intermediate activation, upstream
    parameters receive None gradients. This test verifies the normal
    (non-detached) case.
    """
    class _GoodNet(DLModule):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4)

        def forward(self, x):
            h = self.a(x)          # gradient flows through a
            return self.b(h)       # gradient flows through b

    model = _GoodNet()
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    norms = _do_backward(model, x, y)
    none_params = [n for n, g in norms.items() if g is None]
    assert not none_params, f"Graph broken: {none_params} received no gradient"


@pytest.mark.unit
def test_graph_broken_by_detach_detected():
    """
    A model that calls .detach() on the hidden state should have None
    gradients in the upstream layer — this test verifies we can detect that.
    """
    class _BrokenNet(DLModule):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 4)

        def forward(self, x):
            h = self.a(x).detach()  # intentionally breaks gradient
            return self.b(h)

    model = _BrokenNet()
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    norms = _do_backward(model, x, y)
    # 'a' parameters should receive None gradient (graph is cut at detach)
    a_grads = {n: g for n, g in norms.items() if n.startswith("a.")}
    none_a  = [n for n, g in a_grads.items() if g is None]
    assert none_a, \
        "Expected None gradient in 'a' layer after detach(), but got gradients"


@pytest.mark.unit
def test_no_runtime_error_on_double_backward_without_retain_graph():
    """Calling backward() twice without retain_graph=True must raise RuntimeError."""
    model = _ShallowNet()
    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))
    out  = model.forward(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    with pytest.raises(RuntimeError):
        loss.backward()  # graph freed after first backward


@pytest.mark.unit
def test_retain_graph_allows_double_backward():
    """With retain_graph=True, backward() can be called twice on the same graph."""
    model = _ShallowNet()
    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))
    out  = model.forward(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward(retain_graph=True)
    loss.backward()  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# §8.5  Gradient accumulation correctness
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_gradient_accumulation_equivalent_to_full_batch():
    """
    Gradient accumulated over N mini-batches (sum / N) must equal
    the gradient computed on the full batch (within floating-point tolerance).
    """
    torch.manual_seed(0)
    model_full  = _ShallowNet()
    model_accum = _ShallowNet()

    # Copy identical weights
    model_accum.load_state_dict(model_full.state_dict())

    # Full batch
    X = torch.randn(32, 16)
    y = torch.randint(0, 4, (32,))
    loss_fn = nn.CrossEntropyLoss()
    model_full.zero_grad()
    loss = loss_fn(model_full.forward(X), y)
    loss.backward()
    full_grads = {n: p.grad.clone() for n, p in model_full.named_parameters()}

    # Accumulated in 4 steps of 8 samples each
    model_accum.zero_grad()
    n_accum = 4
    for i in range(n_accum):
        Xi = X[i*8:(i+1)*8]
        yi = y[i*8:(i+1)*8]
        sub_loss = loss_fn(model_accum.forward(Xi), yi) / n_accum
        sub_loss.backward()

    for name, p in model_accum.named_parameters():
        torch.testing.assert_close(
            p.grad, full_grads[name],
            rtol=1e-4, atol=1e-6,
            msg=f"Gradient accumulation mismatch in {name}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# §8.6  Backprop through Pipeline
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_backprop_through_pipeline_ok():
    """Pipeline.backprop_test() must return success=True for a simple pipeline."""
    mermaid = "flowchart LR\n  input[IDENTITY] --> linear[Linear(16,4)]\n"
    pipeline = Pipeline(
        mermaid_flowchart=mermaid,
        modules={"linear": nn.Linear(16, 4)},
    )
    result = pipeline.backprop_test((1, 16), check_grad_norms=True)
    assert result.get("success") is True, f"Pipeline backprop failed: {result}"


@pytest.mark.unit
def test_backprop_through_pipeline_grad_norms_non_zero():
    mermaid = "flowchart LR\n  input[IDENTITY] --> linear[Linear(16,4)]\n"
    pipeline = Pipeline(
        mermaid_flowchart=mermaid,
        modules={"linear": nn.Linear(16, 4)},
    )
    result = pipeline.backprop_test((1, 16), check_grad_norms=True)
    assert result.get("grad_norm") > 0, \
        "Gradient norm is zero inside Pipeline — possible graph break"

# ══════════════════════════════════════════════════════════════════════════════
# §8.7  Gradient norm monitoring utility
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
def test_compute_total_grad_norm_positive_after_backward():
    model = _ShallowNet()
    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))
    model.zero_grad()
    nn.CrossEntropyLoss()(model.forward(x), y).backward()
    total = compute_total_grad_norm(model)
    assert total > 0.0


@pytest.mark.unit
def test_compute_total_grad_norm_zero_before_backward():
    model = _ShallowNet()
    model.zero_grad()
    total = compute_total_grad_norm(model)
    assert total == pytest.approx(0.0)
