---
title: Precision & Quantization
type: docs
sidebar:
  open: true
weight: 400
---



## 1. What “Precision” Actually Means

**Precision** refers to the **numerical format used to represent and compute tensors**.

Common floating-point formats:

* `float64` (FP64, double precision)
* `float32` (FP32, single precision)
* `float16` (FP16, half precision)
* `bfloat16` (BF16, brain floating point, better floating point compared to FP16)

Key properties of a floating-point format:

* **Exponent range** → how large/small numbers can be
* **Mantissa bits** → how precise numbers are

![tensor dtypes](/precision-and-quantization/tensor-dtypes.png)

> [!NOTE]
>
> `torch.float16` and `torch.bfloat16` both use 16 bits but differ in exponent/mantissa layout.
>
> - **`BF16`** shares FP32’s exponent range, so it is typically `more numerically stable`.
> - **`FP16`**’s smaller exponent range makes it more prone to `overflow/underflow` with very large or small values.

Important distinction:

> Precision applies to **computation**, not just storage.

In real training systems, **not all tensors share the same precision**.

Typical setup:

* Weights: FP16 or BF16
* Activations: FP16 or BF16
* Gradients: FP16 (sometimes accumulated in FP32)
* Optimizer states: FP32

So “FP16 training” does **not** mean “everything is FP16”.

---

## 2. Why Lower Precision Is Faster

Lower precision helps because:

* Less memory bandwidth
* Better cache utilization
* Specialized hardware units (Tensor Cores)

On modern GPUs:

* FP16 / BF16 matmuls are **much faster** than FP32
* Some ops are numerically unsafe in low precision

This creates a tension:

* Speed wants low precision
* Stability wants high precision

Mixed precision exists to resolve this tension.

---

## 3. Automatic Mixed Precision (AMP)

**Mixed precision training** uses *multiple* numerical formats during training.

Core idea:

* Use low precision where it is safe
* Use high precision where it is necessary

In PyTorch, AMP is implemented via:

* `torch.autocast`
* `GradScaler`

### What AMP Does

Inside an autocast region:

* Matrix multiplications → FP16 or BF16
* Convolutions → FP16 or BF16
* Reductions, normalizations → FP32
* Loss computation → FP32

Crucially:

* **Master weights are kept in FP32**
* Forward pass may use FP16/BF16 copies
* Gradients are scaled to avoid underflow

### Loss Scaling

FP16 has limited exponent range → small gradients may underflow to zero.

Solution:

* Multiply loss by a large scale factor
* Backpropagate
* Divide gradients by scale before optimizer step

This is handled automatically by `GradScaler`.

---

## 4. FP16 vs BF16

These are not interchangeable.

### FP16

* Smaller exponent
* Higher risk of overflow/underflow
* Needs loss scaling
* Faster on older GPUs

### BF16

* Same exponent range as FP32
* Lower mantissa precision
* Much more numerically stable
* Preferred on A100 / H100

Rule of thumb:

* If BF16 is supported → use BF16
* Otherwise → FP16 + AMP

---

## 5. Precision vs Casting vs Quantization

Important separation:

* **Casting**: changing floating-point format
  Example: FP32 → FP16

* **Precision**: how computations are executed
  Example: matmul in FP16, accumulation in FP32

* **Quantization**: mapping floating-point values to a **discrete integer set**

FP32 → FP16 is **not quantization**.
FP32 → INT8 **is quantization**.

---

## 6. What Quantization Is

**Quantization** maps real values to a finite set of integers.

General idea:

* Continuous values → discrete bins
* Requires scale and (optionally) zero-point
* Loses information irreversibly

Typical goals:

* Reduce memory footprint
* Speed up inference
* Enable deployment on constrained hardware

Quantization is primarily an **inference-time optimization**, though training-time variants exist.

---

## 7. Affine Quantization (Scale + Zero Point)

Given:

* Floating-point range: `[x_min, x_max]`
* Integer range: `[q_min, q_max]`

### Parameters

```
scale = (x_max - x_min) / (q_max - q_min)
zero_point = q_min - round(x_min / scale)
```

### Quantization

```
q = clamp(round(x / scale + zero_point), q_min, q_max)
```

### Dequantization

```
x̂ = (q - zero_point) * scale
```

This is called **affine quantization**.

---

## 8. Symmetric Quantization

A common simplification:

* Integer range is symmetric (e.g. [-127, 127])
* Zero point = 0
* Only scale is used

Benefits:

* Simpler math
* Faster kernels
* Common in deep learning accelerators

Tradeoff:

* Slightly worse accuracy than affine quantization

---

## 9. Post-Training Quantization (PTQ)

Workflow:

1. Train model in FP32 (or mixed precision)
2. Freeze weights
3. Quantize weights (and sometimes activations)
4. Run inference

Variants:

* Dynamic quantization
* Static quantization (with calibration data)

PTQ is simple but may degrade accuracy for sensitive models.

---

## 10. Quantization-Aware Training (QAT)

QAT simulates quantization during training.

Mechanism:

* Insert **fake quantization** ops in forward pass
* Apply rounding + clamping
* Backward pass uses **straight-through estimator (STE)**

Key idea:

* Model learns weights that are robust to quantization
* Gradients ignore non-differentiable rounding

QAT improves accuracy at the cost of training complexity.

---

## 11. Mental Model Summary

* Precision → **how math is computed**
* AMP → **dynamic choice of precision**
* Casting → **floating-point format change**
* Quantization → **float → integer mapping**
* PTQ → quantize after training
* QAT → train with quantization effects simulated

---
