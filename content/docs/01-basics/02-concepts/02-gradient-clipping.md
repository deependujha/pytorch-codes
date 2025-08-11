---
title: Gradient Clipping
type: docs
prev: docs/01-basics/02-concepts/01-learning-rate-scheduler.md
sidebar:
  open: false
weight: 23
---

## Correct way to do gradient clipping

- after `loss.backward()` & before `optimizer.step()`

```python
optimizer.zero_grad()        
loss, hidden = model(data, hidden, targets)
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
```

