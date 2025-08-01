# Gradient Clipping

## Correct way to do gradient clipping

- after `loss.backward()` & before `optimizer.step()`

```python
optimizer.zero_grad()        
loss, hidden = model(data, hidden, targets)
loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
```

