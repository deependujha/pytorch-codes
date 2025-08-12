---
title: Learning Rate Scheduler
type: docs
prev: docs/01-basics/02-concepts/01-tensor.md
next: docs/01-basics/02-concepts/03-gradient-clipping.md
sidebar:
  open: false
weight: 23
---

A **Learning Rate Scheduler** adjusts the optimizer’s LR during training for better convergence.


## Usage Order in Training Loop

### **Most schedulers** (StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR…)

Call **after** the optimizer updates the weights:

```python
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()       # update model params
    scheduler.step()           # update LR for next epoch/step
```

You can also step **every batch** instead of every epoch, depending on scheduler type and your decay strategy.

---

### **ReduceLROnPlateau**

Metric-based — must be called **after validation**, not after optimizer step:

```python
for epoch in range(epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)   # monitors metric and adjusts LR
```

---

## Common Schedulers

- **StepLR** – drop LR by `gamma` every `step_size` epochs

`lr_t = lr_0 × gamma^(⌊epoch / step_size⌋)`

```python
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
```

- **MultiStepLR** – drop LR at specific epochs in `milestones`

`lr_t = lr_prev × gamma` if `epoch ∈ milestones`

```python
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
```

- **ExponentialLR** – multiply LR by `gamma` every step/epoch

`lr_t = lr_0 × gamma^t`

```python
scheduler = ExponentialLR(optimizer, gamma=0.95)
```

- **CosineAnnealingLR** – cosine decay from `lr_max` to `lr_min` over `T_max` steps

`lr_t = lr_min + 0.5 × (lr_max − lr_min) × (1 + cos(π × t / T_max))`

```python
scheduler = CosineAnnealingLR(optimizer, T_max=50)
```

- **ReduceLROnPlateau** – reduce LR by `factor` after `patience` epochs without metric improvement

`lr_new = lr_prev × factor`

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
```

---

## Tips

* Step **after optimizer.step()** unless using `ReduceLROnPlateau`.
* Track LR during training:

  ```python
  current_lr = optimizer.param_groups[0]['lr']
  ```

* Batch-level scheduling reacts faster; epoch-level scheduling is smoother.
