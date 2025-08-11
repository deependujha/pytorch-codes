---
title: Tensors in PyTorch
type: docs
prev: docs/01-basics/01-data/01-dataset-and-dataloader.md
next: docs/01-basics/02-concepts/02-learning-rate-scheduler.md
sidebar:
  open: false
weight: 22
---

## Creation

```python
import torch

# from array
torch.tensor([1, 2, 3])
torch.tensor([[1, 2], [3, 4]])

# Random
torch.rand(2, 3)        # Uniform [0, 1)
torch.randn(2, 3)       # Normal (mean=0, std=1)

torch.empty(size)       # uninitialized memory (fast but contains garbage values).

# torch.randint(low, high, shape)
torch.randint(0, 10, (2, 3))  # Integers [0, 10)

# Constants
torch.zeros(2, 3) # shape
torch.ones(2, 3)

# torch.full(size, fill_value)
torch.full((2, 3), 7) # tensor of shape (2, 3) initialized with 7

# Sequences
torch.arange(0, 10, 2)     # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5)    # 5 points between 0 and 1
torch.eye(3)               # Identity matrix
```

### `torch.empty` v/s `torch.rand`

`torch.empty` and `torch.rand` are very different beasts even though both give you a tensor “full” of numbers.

* **`torch.empty(shape)`**

  * Allocates memory **without** initializing it.
  * The contents are whatever random bits happened to be in RAM/VRAM — garbage values.
  * Very fast because it skips filling the tensor.
  * Useful if you plan to immediately overwrite all elements.
  * Dangerous if you accidentally use it before assignment (can produce NaNs, huge values, etc.).

* **`torch.rand(shape)`**

  * Allocates **and** fills with random values from **uniform distribution \[0, 1)**.
  * Safe to use directly in computations.
  * Slightly slower because it does actual initialization.

Example:

```python
torch.empty(2, 3)
# tensor([[ 5.1396e-02,  0.0000e+00,  3.9894e-08],
#         [ 0.0000e+00,  1.2867e+31, -1.0094e+25]])  # garbage data

torch.rand(2, 3)
# tensor([[0.4728, 0.7289, 0.1710],
#         [0.9804, 0.3112, 0.6945]])  # nice [0, 1) random numbers
```

Rule of thumb:
Use `empty` only if you **know** you’ll overwrite everything before reading. Otherwise, `rand` (or `zeros`/`ones`) is the safe choice.

---

## Properties

```python
t = torch.randn(2, 3)
t.shape        # torch.Size([2, 3])
t.dtype        # e.g. torch.float32
t.device       # 'cpu' or 'cuda:0'
```

## Device/Dtype

```python
t = torch.ones(2, 3, device='cuda', dtype=torch.float64)
t_cpu = t.to('cpu')
t_float = t.to(torch.float32)
```

## Indexing & Slicing

```python
t[0]          # First row
t[:, 1]       # Second column
t[0, 1]       # Single element
```

## Reshape

```python
t.view(6)                 # Flatten
t.view(3, 2)               # New shape
t.permute(1, 0)            # Swap dimensions
t.unsqueeze(0)             # Add dim at pos 0
t.squeeze()                # Remove dims of size 1
```

## Math Ops

```python
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
a + b
a * b                      # Elementwise
torch.matmul(a.view(1, -1), b.view(-1, 1))  # Matrix mult
```

## Interop

```python
import numpy as np

np_arr = np.array([1, 2])
t = torch.from_numpy(np_arr)
np_back = t.numpy()
```

## Saving/Loading

```python
torch.save(t, 'tensor.pt')
t_loaded = torch.load('tensor.pt')
```

## TensorDataset

In PyTorch, `TensorDataset` is a simple dataset wrapper from `torch.utils.data` that stores one or more tensors of the same length and lets you index them together.

Think of it as a zip for tensors — you give it, say, your inputs and labels, and it returns matching slices when you index it.

Example:

```python
from torch.utils.data import TensorDataset, DataLoader
import torch

# Two tensors of equal length
features = torch.randn(5, 3)  # 5 samples, 3 features each
labels = torch.randint(0, 2, (5,))  # 5 labels

dataset = TensorDataset(features, labels)

# Indexing returns a tuple
x, y = dataset[0]
print(x, y)

# You can wrap it in a DataLoader for batching
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch_x, batch_y in loader:
    print(batch_x, batch_y)
```

Key points:

* All tensors passed in must have the same first dimension size (number of samples).
* Useful when your entire dataset already fits in memory as tensors.
* Minimal — no transformations, lazy loading, or file handling; just indexing and length.

It’s perfect for small to medium datasets, prototyping, or unit tests — but for large datasets, you’d usually subclass `Dataset` to load data on the fly instead of keeping it all in memory.
