---
title: Module vs Parameter vs Buffer
type: docs
sidebar:
  open: false
weight: 1
math: true
---

## Core Idea

PyTorch separates **computation**, **state**, and **optimization** on purpose.

* **Modules** define computation and own state
* **Parameters** are learnable state (optimized by SGD)
* **Buffers** are persistent state (not optimized)

---

## `nn.Parameter`

**What it is**

* A `torch.Tensor` wrapped with *optimization intent*
* Must be **registered inside a Module** to matter

**Key properties**

* `requires_grad = True`
* Appears in `model.parameters()`
* Saved in `state_dict`
* Updated by **optimizer**, not by `forward`

**Important nuance**

* `requires_grad=True` alone is **not enough**
* Plain tensors with gradients are ignored by optimizers unless they are `nn.Parameter`

**Mental rule**

> “Is the optimizer allowed to change this?”

---

## Buffers (`register_buffer`)

**What it is**

* A tensor that is **part of model state**, but **not learnable**

**Key properties**

* `requires_grad = False`
* Does **not** appear in `model.parameters()`
* **Does** appear in `state_dict`
* Moves with `.to(device)`

**Critical nuance**

* Buffers **can change during training**
* They are updated by **logic**, not gradients

**Canonical examples**

* BatchNorm `running_mean`, `running_var`
* Masks, counters, EMA statistics

**Mental rule**

> “Should this persist and move with the model, but never get gradients?”

---

## `nn.Module`

**What it is**

* A **container + computation definition**
* The unit of composition in PyTorch

**Responsibilities**

* Defines `forward()` (pure computation)
* Registers parameters, buffers, submodules
* Handles:

  * device movement (`.to`)
  * mode switching (`.train() / .eval()`)
  * serialization (`state_dict`)
  * recursion (`modules()`, `parameters()`)

**What it does NOT do**

* Does not update parameters
* Does not run optimization
* Does not “train” by itself

Training happens **outside**:

```python
loss.backward()
optimizer.step()
```

**Mental rule**

> “Does this define structure or computation?”

---

## Side-by-Side Summary

| Concept   | Learnable | Gradients | Optimizer sees | Saved | Moves device |
| --------- | --------- | --------- | -------------- | ----- | ------------ |
| Parameter | yes       | yes       | yes            | yes   | yes          |
| Buffer    | no        | no        | no             | yes   | yes          |
| Tensor    | maybe     | maybe     | no             | no    | no           |
| Module    | n/a       | n/a       | n/a            | n/a   | via contents |

---

## Common Failure Modes (Practical Warnings)

* Using plain tensors instead of `nn.Parameter` → optimizer silently ignores them
* Forgetting `register_buffer` → state doesn’t move or checkpoint
* Updating parameters inside `forward` → breaks autograd, compilation, DDP
* Confusing `train()/eval()` with “on/off gradients” → they mainly affect **buffers**

---

## Code

```python
import torch
import torch.nn as nn


class Core(nn.Module):
    def __init__(self):
        super(Core, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.p1 = nn.Parameter(torch.ones(2))
        self.register_buffer("c1", torch.zeros(2))
        self.register_buffer("c2", torch.ones(2))

    def forward(self):
        raise NotImplementedError("This is just a dummy module for testing purposes.")


if __name__ == "__main__":
    model = Core()

    print(f" Model ".center(50, "-"))
    print(model)
    print()  # Just for better readability

    print(f" Parameters ".center(50, "-"))
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
    print()  # Just for better readability

    print(f" Buffers ".center(50, "-"))
    for name, buffer in model.named_buffers():
        print(f"{name}: {buffer}")
    print()  # Just for better readability

    print(" all parameters in list (used by optimizer) ".center(50, "-"))
    print(list(model.parameters()))
    print()  # Just for better readability
    
    print(" all buffers in list (used by state_dict) ".center(50, "-"))
    print(list(model.buffers()))
    print()  # Just for better readability
```

<details>
  <summary>output</summary>

```md
--------------------- Model ----------------------
Core(
  (l1): Linear(in_features=2, out_features=2, bias=True)
)

------------------- Parameters -------------------
p1: Parameter containing:
tensor([1., 1.], requires_grad=True)
l1.weight: Parameter containing:
tensor([[ 0.7058, -0.2758],
        [ 0.0402, -0.0530]], requires_grad=True)
l1.bias: Parameter containing:
tensor([0.2554, 0.0756], requires_grad=True)

-------------------- Buffers ---------------------
c1: tensor([0., 0.])
c2: tensor([1., 1.])

--- all parameters in list (used by optimizer) ---
[Parameter containing:
tensor([1., 1.], requires_grad=True), Parameter containing:
tensor([[ 0.7058, -0.2758],
        [ 0.0402, -0.0530]], requires_grad=True), Parameter containing:
tensor([0.2554, 0.0756], requires_grad=True)]

---- all buffers in list (used by state_dict) ----
[tensor([0., 0.]), tensor([1., 1.])]
```
</details>

## One-Line Cheatsheet

* **Parameter** → optimized state
* **Buffer** → persistent state
* **Module** → computation + ownership

This trio is the foundation beneath DDP, FSDP, checkpointing, and `torch.compile`. If this model is clean in your head, the rest of PyTorch stops feeling magical and starts feeling mechanical—in a good way.
