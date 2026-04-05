# Usage Guide

## The Factory Pattern

The core of the library is the `Activation` class. It serves as a universal factory.

```python
from Code.models.deep_learning.activations.ActivationFunction import Activation

# Basic
model = Activation("gelu")

# Debug Mode (Enables NaN/Inf checks)
model = Activation("gelu", debug=True)
```

## Family-Specific Usage

### 1. Adaptive Activations

Adaptive activations contain learnable parameters. You usually need to provide the number of channels (`dims`) or specific configuration dictionaries. The library uses a smart broadcasting mechanism (`_smart_broadcast`) that supports `(N, C, H, W)` tensors automatically.

```python
# PReLU-style (Learnable alpha per channel)
# Pass 'dims' matching your channel count (C)
model = Activation("parametric-prelu", dims=(64,))

# EIS (Exponential Integration Sigmoid) - Highly parameteric
# Pass 'dims' for channel-wise adaptation
model = Activation("eis-1", dims=(64,))
```

### 2. Complex Activations

Handling complex-valued data requires specifying an operation `mode`. The wrapper robustly handles these internal conversions.

- `sep`: Apply activation separately to Real and Imaginary parts.
- `mag`: Apply activation to the magnitude, preserving phase.
- `rot`: Apply activation after a learnable rotation.
- `complex`: Use specialized complex-valued functions (e.g. `ComplexReLU`).

```python
x_complex = torch.complex(torch.randn(1, 64), torch.randn(1, 64))

# Separate mode (Default)
act = Activation("complex-relu", mode='sep')

# Rotation mode (Requires dims for phase parameter)
act_rot = Activation("complex-relu", mode='rot', dims=(64,))

# Magnitude mode
act_mag = Activation("complex-relu", mode='mag')
```

### 3. Parametric (Non-Adaptive)

Even "non-adaptive" families often have parametric variants (e.g. `ParametricSoftplus`). These follow the same `dims` pattern if they have learnable implementation.

```python
# Parametric Softplus
model = Activation("parametric-softplus", dims=(64,))
```

## Advanced Features

### Custom String Activations

You can define activations on the fly using string math formulas.

```python
# Define a custom Gated Linear Unit variant
# x * sigmoid(w1 * x + b1)
model = Activation("w1*x*sigmoid(x)", in_features=64)
```

### Registry & Extensibility

The library maintains a JSON registry `src/Activations/__registry__.json` mapping names to file paths. This allows lazy loading.

- **Lazy Loading**: Modules are only imported when requested.
- **Debug**: Use `Activation._name_to_info` to see what has been loaded.
