---
title: Activation Memory
type: docs
sidebar:
  open: false
weight: 27
math: true
---

**Activation memory** is the memory used to store intermediate results produced during the forward pass that are required to compute gradients during the backward pass.

Intuitively, training a neural network is a **reversible computation**: the backward pass must undo the forward pass. Activation memory exists to make this reversal possible.

---

## A minimal example

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 5
loss = z ** 2
loss.backward()
```

During the forward pass:

* `y` and `z` are computed as intermediate values.

During the backward pass:

* To compute `∂loss/∂z`, the value of `z` is required.
* To compute `∂loss/∂x`, information from `y = x * 3` is required.

Because these values are needed later, PyTorch saves them during the forward pass.
These saved tensors constitute **activation memory**.

---

## Why backward needs forward values

Backpropagation proceeds **operation by operation**, in reverse order.
Each operation computes gradients using **locally available information**, not global formulas.

For example:

* For `z = y + 5`, the gradient is constant.
* For `loss = z²`, the gradient is `2z`, which requires knowing `z`.

Backward cannot reconstruct these values symbolically; it must either:

1. Load them from memory, or
2. Recompute them by rerunning part of the forward pass.

By default, PyTorch chooses (1).

---

## Activation memory in neural networks

Consider a common pattern:

```python
h = x @ W + b
a = torch.relu(h)
```

To compute gradients:

* The matrix multiplication needs `x`.
* ReLU needs to know where `h > 0`.

As a result, PyTorch stores:

* `x`
* `h` (or a mask derived from it)

In deep networks, this happens for every layer and every batch element, which is why activation memory often exceeds parameter memory.

---

## What activation memory is not

Activation memory is distinct from:

* **Parameters** (model weights)
* **Gradients** (`param.grad`)
* **Optimizer state** (e.g. Adam momentum and variance)

Among these, activation memory is usually the largest contributor during training.

---

## Lifetime of activation memory

* Activation memory is allocated during the forward pass.
* It is released after the backward pass completes.
* It is not used during inference unless gradients are enabled.

Using `torch.no_grad()` disables activation storage entirely.

---

## Reducing activation memory: checkpointing

Activation checkpointing reduces memory usage by **not saving all activations**.

Instead:

* Only selected **checkpoint boundaries** are saved.
* Intermediate activations between checkpoints are discarded.
* During backward, the forward pass is re-executed between checkpoints to regenerate missing values.

Example (conceptual):

```python
# checkpoint
b = a * 3

# not stored
c = b + 5
d = c * 3
e = d + 5

# checkpoint
loss = e ** 2
```

During backward:

* `e` is available.
* `c` and `d` are recomputed by rerunning the forward pass starting from `b`.

This trades:

* **Lower memory usage**
* **Higher computation cost**

---

## Why recomputation is valid

Checkpointing relies on forward computation being:

* Deterministic
* Pure (no side effects)
* Replayable

Under these conditions, recomputed activations are identical to the originals.

---

## Why activation memory dominates training cost

Activation memory:

* Scales with batch size
* Scales with network depth
* Exists only during training
* Is required even when parameters are frozen

For large models, activation memory is often the primary reason for out-of-memory errors.

---

## Summary

* Activation memory stores forward-pass intermediates required for backpropagation
* Backward computation is local and numerical, not symbolic
* If a value is needed for gradient computation, it must be saved or recomputed
* Activation checkpointing reduces memory by recomputing forward segments
* Understanding activation memory is critical for training large and deep models
