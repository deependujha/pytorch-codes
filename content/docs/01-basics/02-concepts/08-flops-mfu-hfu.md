---
title: FLOPs, MFU, and HFU
type: docs
sidebar:
  open: false
weight: 28
math: true
---

## FLOPs in training (intuition)

Training a neural network performs a large number of floating-point operations (FLOPs). These operations come from:

* Forward passes through the model
* Backward passes to compute gradients
* Optional recomputation of activations

FLOPs provide a convenient abstraction for reasoning about **compute cost**, even though they do not directly measure runtime.

A single training step consists of:

* Forward pass
* Backward pass

A common approximation:

```
Backward FLOPs ≈ 2 × Forward FLOPs
```

Therefore:

```
Training FLOPs ≈ 3 × Forward FLOPs
```

This approximation is widely used for estimating **model-required FLOPs**.

---

## Why backward FLOPs are almost twice forward FLOPs

### Forward vs backward computation

The forward pass computes the output of each layer:

\[
y = f(x, W)
\]

The backward pass computes **two distinct quantities**:

1. Gradients with respect to the parameters
   \[
   \frac{\partial L}{\partial W}
   \]

2. Gradients with respect to the layer inputs
   \[
   \frac{\partial L}{\partial x}
   \]

Both are required:

* Parameter gradients are used by the optimizer.
* Input gradients propagate learning to earlier layers.

---

### Why we compute both gradients

The backward pass must compute both gradients because:

* Parameter gradients are necessary for updating the model weights.
* Input gradients are necessary for propagating the learning signal backward through the network.

```python
# x → Linear1 → Linear2 → loss
h = Linear1(x)
y = Linear2(h)
loss = L(y)
```

#### For `Linear2`

* Needs `∂L/∂W2` → update weights
* Needs `∂L/∂h` → pass gradient backward so `Linear1` can compute its gradients

That `∂L/∂h` is the **input gradient of Linear2**.

#### For `Linear1`

* Needs `∂L/∂W1` (to update weights)
* Needs `∂L/∂x` (often unused, but still computed)

But to compute `∂L/∂W1`, `Linear1` needs `∂L/∂h`.

Where did that come from?

👉 **From Linear2’s input gradient**

---

### The chain rule, operationally

Mathematically:

\[
\frac{∂L}{∂W_1} =
\frac{∂L}{∂h}
\cdot
\frac{∂h}{∂W_1}
\]

If `∂L/∂h` is missing, `∂L/∂W1` is impossible.

---

### Example: linear layer

Consider a linear layer:

```python
y = x @ W
```

where:

* `x` has shape `(B, D)`
* `W` has shape `(D, H)`
* `y` has shape `(B, H)`

#### Forward FLOPs

Matrix multiplication requires:

\[
\text{FLOPs}_{\text{forward}} = 2 \times B \times D \times H
\]

> In matrix multiplication, each output element requires `D` multiplications and `D−1` additions, which is approximately `2D` FLOPs per output element.

---

#### Backward FLOPs

The backward pass performs two matrix multiplications:

1. Gradient with respect to weights:
   \[
   \frac{\partial L}{\partial W} = x^T @ \frac{\partial L}{\partial y}
   \]

2. Gradient with respect to inputs:
   \[
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} @ W^T
   \]

Each has similar cost to the forward pass, giving:

\[
\text{FLOPs}*{\text{backward}} \approx 2 \times \text{FLOPs}*{\text{forward}}
\]

---

### Why this generalizes

Most trainable layers (linear, convolution, attention, MLPs) follow the same pattern:

* Forward computes outputs
* Backward computes gradients for both parameters and inputs

Elementwise nonlinearities (e.g. ReLU) have low FLOP cost and do not significantly affect the overall ratio, which is dominated by matrix multiplications.

---

### Implication for FLOPs-based metrics

Because backward FLOPs dominate:

* Model-required training FLOPs are often approximated as:
  \[
  \text{Training FLOPs} \approx 3 \times \text{Forward FLOPs}
  \]
* This approximation is used when computing **MFU**
* Additional recomputation increases **hardware-executed FLOPs (HFU)** but does not change **model-required FLOPs (MFU)**

---

## Two different notions of FLOPs

There are two distinct FLOP counts involved in training.

### Model-required FLOPs

The minimum number of FLOPs mathematically required to train the model:

* Forward pass
* Backward pass
* No recomputation

This depends only on:

* Model architecture
* Batch size
* Sequence length

It is independent of how training is implemented.

---

### Hardware-executed FLOPs

The actual number of FLOPs executed by the accelerator:

* Forward pass
* Backward pass
* **Plus any recomputation**

This depends on:

* Activation checkpointing
* Kernel fusion
* Training strategy

Hardware-executed FLOPs are always **greater than or equal to** model-required FLOPs.

---

## Model FLOPs Utilization (MFU)

**MFU measures how efficiently the model’s required computation is executed on the hardware.**

\[
\text{MFU} =
\frac{\text{model-required FLOPs per second}}
{\text{theoretical peak hardware FLOPS}}
\]

### Intuition

MFU answers the question:

> *Given the math this model must perform, how close are we to the hardware’s peak compute capability?*

MFU focuses on **useful work** and ignores:

* Recomputation
* Redundant execution
* Training tricks

---

### What MFU reflects

MFU is sensitive to:

* Batch size
* Sequence length
* Tensor shapes
* Kernel efficiency
* Parallelism

MFU improves when the model’s computation maps better to the hardware.

---

### What MFU does not reflect

MFU does **not** change with:

* Activation checkpointing
* Recomputation
* Extra redundant work

Because these do not change the model’s mathematical requirements.

---

## Hardware FLOPs Utilization (HFU)

**HFU measures how efficiently the hardware’s compute capacity is being used.**

\[
\text{HFU} =
\frac{\text{hardware-executed FLOPs per second}}
{\text{theoretical peak hardware FLOPS}}
\]

### Intuition

HFU answers the question:

> *How busy was the accelerator during training?*

HFU includes:

* Forward computation
* Backward computation
* **Recomputation**

---

### What HFU reflects

HFU increases when:

* More computation is executed per second
* Recomputation is introduced
* Memory stalls are reduced by trading memory for compute

A higher HFU means the hardware is closer to saturation.

---

### What HFU does not reflect

HFU does **not** indicate:

* Whether the computation was necessary
* Whether training is faster
* Whether the model is efficient

HFU can increase even when redundant work is added.

---

## Recomputation and utilization metrics

Activation recomputation:

* Increases hardware-executed FLOPs
* Reduces memory traffic
* Often improves wall-clock performance

As a result:

* **HFU increases**
* **MFU remains unchanged**

This separation allows MFU to compare models fairly, while HFU explains hardware behavior.

---

### Simple example

Assume:

* GPU peak = 100 TFLOPS
* Model forward + backward = 50 TFLOPs per step
* Step time = 1 second

#### Case A: no recomputation

* Actual FLOPs executed = 50
* HFU = 50 / 100 = 0.5
* MFU = 50 / 100 = 0.5

#### Case B: with recomputation

* Actual FLOPs executed = 80 (`+30 from activation checkpointing recomputation`)
* HFU = 80 / 100 = 0.8
* MFU = 50 / 100 = 0.5

Same model. Same progress. Different hardware usage.

> **HFU measures how much the GPU worked. MFU measures how much useful model work you got.**

---

## Why both metrics exist

| Metric | Purpose                   |
| ------ | ------------------------- |
| MFU    | Model–hardware efficiency |
| HFU    | Hardware utilization      |

MFU is useful for:

* Comparing model implementations
* Reasoning about architectural efficiency

HFU is useful for:

* Diagnosing hardware underutilization
* Understanding memory vs compute bottlenecks

---

## Important caveat

Neither MFU nor HFU directly measure training speed.

* Higher MFU does not guarantee faster training
* Higher HFU does not guarantee useful work

**Wall-clock time remains the ultimate metric.**

MFU and HFU are diagnostic tools, not optimization targets.

---

## Summary

* FLOPs abstract computational work, not runtime
* Model-required FLOPs and hardware-executed FLOPs are distinct
* MFU measures efficiency of required computation
* HFU measures how busy the hardware was
* Recomputation increases HFU but not MFU
* Both metrics explain performance; neither defines it
