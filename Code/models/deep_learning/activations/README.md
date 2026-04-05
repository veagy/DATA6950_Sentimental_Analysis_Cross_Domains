# src.Activations: Advanced Activation Function Library

## Overview

`src.Activations` is a comprehensive PyTorch library containing over 700+ activation functions, ranging from standard non-adaptive functions to highly experimental adaptive, complex-valued, and gated linear unit variants. It is designed for researchers and advanced practitioners who need fine-grained control over activation dynamics.

## Key Families

- **Non-Adaptive**: Standard activations (ReLU, Sigmoid) and their parametric variants (PReLU, swish) with NCHW broadcasting support.
- **Adaptive**: Learnable activation functions (e.g., `AdaptiveReLU`, `EIS`, `FractionallyAdaptive`) that evolve their shape during training.
- **Complex**: Robust support for complex-valued inputs with multiple operation modes (magnitude, rotation, separate).
- **Gated Linear Units (GLU)**: Extensive collection of GLU variants.

## Quick Start

The primary entry point is the `Activation` factory, which handles registry lookups and instantiation.

```python
import torch
from Code.models.deep_learning.activations.ActivationFunction import Activation

# 1. Standard Activation
act = Activation("relu")
x = torch.randn(1, 64, 32, 32)
out = act(x)

# 2. Adaptive Activation (requires channel dimension)
# Automatically handles NCHW broadcasting
act_adaptive = Activation("parametric-prelu", dims=(64,))
out_adaptive = act_adaptive(x)
```

## Installation

Ensure the parent directory of `src` is in your `PYTHONPATH`. The library is structured as a standard Python package.
