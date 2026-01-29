---
title: Gradient Accumulation
type: docs
prev: docs/01-basics/02-concepts/04-transfer-learning.md
next: docs/01-basics/02-concepts/06-mixed-precision-training.md
sidebar:
  open: false
weight: 25
math: true
---

Gradient accumulation is a technique used to effectively increase the batch size during training in PyTorch, particularly when hardware constraints (like GPU memory) prevent using a large batch size in a single forward-backward pass.

> It’s especially useful in distributed settings where multiple GPUs or nodes are involved.

If we do optimizer step on each forward pass in distributed training, all gradients need to be communicated, increasing communication overhead. Gradient accumulation allows us to accumulate gradients over several forward passes before performing an optimizer step, reducing the frequency of communication.

- Gradient accumulation allows you to simulate a larger batch size by splitting a large batch into smaller mini-batches, computing gradients for each mini-batch, and accumulating them before performing a single optimization step.

- This is useful when:
    - Your GPU memory can’t handle a large batch size.
    - You want to maintain the benefits of larger batch sizes (e.g., more stable gradients, better generalization) without requiring more memory.

---

## How it works?

- Divide the desired batch size into smaller mini-batches.
- For each mini-batch, compute the loss and gradients, but don’t update the model parameters immediately.
- Accumulate (sum) the gradients over multiple mini-batches.
- Once the equivalent of the desired batch size is reached, perform a single optimization step using the accumulated gradients.

---

## Single device Gradient Accumulation

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model, loss function, and optimizer
model = nn.Linear(10, 1).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Hyperparameters
desired_batch_size = 64
accumulation_steps = 4  # Number of mini-batches to accumulate
mini_batch_size = desired_batch_size // accumulation_steps  # 16

# Dummy data (replace with your dataset)
data = torch.randn(64, 10).cuda()
targets = torch.randn(64, 1).cuda()

# Training loop
model.train()
optimizer.zero_grad()  # Clear gradients at the start

for i in range(0, len(data), mini_batch_size):
    # Get mini-batch
    inputs = data[i:i + mini_batch_size]
    target = targets[i:i + mini_batch_size]
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, target)
    
    # Scale loss to simulate full batch (optional, mimics averaging)
    loss = loss / accumulation_steps
    
    # Backward pass (accumulate gradients)
    loss.backward()
    
    # Perform optimization step after accumulating enough gradients
    if (i + mini_batch_size) % desired_batch_size == 0:
        optimizer.step()  # Update model parameters
        optimizer.zero_grad()  # Clear gradients for next accumulation
        print(f"Optimization step performed after {accumulation_steps} mini-batches")

# Note: If len(data) % desired_batch_size != 0, handle the remaining data
```

---

## Why Do We Scale the Loss in Gradient Accumulation?

Gradient accumulation simulates a larger batch size by splitting it into `n` smaller microbatches and delaying the optimizer step until all microbatches are processed.

Most training setups use **mean reduction** for loss (e.g., `reduction="mean"`). Understanding this is key to why loss scaling is needed.

---

### What a True Large Batch Computes

Assume:
* Microbatch size = `m`
* Number of accumulation steps = `n`
* Total batch size = `n × m`

**For a true large batch**, the loss is the average over all samples:

\[
L_{\text{large}} = \frac{\text{loss}_1 + \text{loss}_2 + \cdots + \text{loss}_{n \times m}}{n \times m}
\]

We can regroup this as an average of microbatch averages:

\[
L_{\text{large}} = \frac{1}{n} \left( \frac{\text{loss}_1 + \cdots + \text{loss}_m}{m} + \frac{\text{loss}_{m+1} + \cdots + \text{loss}_{2m}}{m} + \cdots \right)
\]

Let's call each microbatch's average loss as `L₁`, `L₂`, ..., `Lₙ`. So:

\[
L_{\text{large}} = \frac{L_1 + L_2 + \cdots + L_n}{n}
\]

---

### What Happens in Gradient Accumulation

During gradient accumulation:
* Each microbatch computes its own mean loss `Lᵢ`
* We call `.backward()` on each `Lᵢ` separately
* Gradients from each backward pass are **summed** (accumulated)
* After `n` accumulations, we do one optimizer step

**Without loss scaling**, the accumulated gradient is:

\[
\nabla L_1 + \nabla L_2 + \cdots + \nabla L_n
\]

But the gradient of the true large batch should be:

\[
\nabla L_{\text{large}} = \nabla \left( \frac{L_1 + L_2 + \cdots + L_n}{n} \right) = \frac{\nabla L_1 + \nabla L_2 + \cdots + \nabla L_n}{n}
\]

**The problem**: Without scaling, our accumulated gradients are **`n` times too large**, which effectively multiplies the learning rate by `n`.

---

### Solution: Scale the Loss

To match the true large batch gradient, divide each loss by `n` before calling backward:

```python
(loss / n).backward()
```

This gives accumulated gradients:

\[
\frac{\nabla L_1}{n} + \frac{\nabla L_2}{n} + \cdots + \frac{\nabla L_n}{n} = \frac{\nabla L_1 + \nabla L_2 + \cdots + \nabla L_n}{n}
\]

This **exactly matches** the gradient from true large-batch training.

---

### Key Takeaway

Loss scaling in gradient accumulation ensures the **gradient magnitude matches** that of a true large batch, preserving the same optimization dynamics and effective learning rate.

---

#### Additional Key Points

- **Loss scaling**: Dividing the loss by `accumulation_steps` ensures the gradients are scaled appropriately, mimicking the effect of averaging gradients over the full batch.
- **Gradient clearing**: Call `optimizer.zero_grad()` only after the optimization step to allow gradients to accumulate across mini-batches.
- **Edge case**: If the dataset size isn't perfectly divisible by `desired_batch_size`, you may need to handle the last incomplete batch separately (e.g., adjust loss scaling or perform an early optimization step).

---

## DDP gradient accumulation

- Rather than communicating gradients after every mini-batch, DDP can accumulate gradients locally on each GPU for several mini-batches before synchronizing.
- This reduces the frequency of communication and can lead to better performance, especially in scenarios with high communication overhead.

Gradient accumulation in PyTorch's DistributedDataParallel (DDP) allows for simulating larger batch sizes than what can fit into a single GPU's memory by accumulating gradients over multiple mini-batches before performing an optimizer step. 
Steps for Gradient Accumulation in DDP: 

- **`Initialize DDP`**: Wrap your model with torch.nn.parallel.DistributedDataParallel. 

```python
    model = DDP(model.to(device), device_ids=[local_rank])
```

- **`Disable Gradient Synchronization for Intermediate Steps`**: For the mini-batches within an accumulation cycle (except the last one), prevent DDP from synchronizing gradients across processes after each loss.backward(). This is crucial to avoid unnecessary communication overhead. Use the no_sync() context manager for this. 

```python
    for i, (inputs, labels) in enumerate(train_loader):
        # ... (data loading and moving to device)

        # Accumulate gradients without synchronization
        if (i + 1) % accumulation_steps != 0:
            with model.no_sync():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
        else:
            # Last mini-batch of the accumulation cycle, synchronize gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
```

- **`Perform Optimizer Step and Zero Gradients`**: Only after accumulating gradients for accumulation_steps mini-batches, perform the optimizer.step() and optimizer.zero_grad(). This ensures that the optimizer updates parameters based on the accumulated gradients from the larger effective batch.

```python
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Explanation

- **`model.no_sync()`**: This context manager temporarily disables the gradient synchronization mechanism within DDP. When `loss.backward()` is called within `no_sync()`, gradients are computed and stored locally on each GPU, but no allreduce operation (which aggregates gradients across all DDP processes) occurs.
- **`Last Mini-batch`**: For the final mini-batch in an accumulation cycle, `no_sync()` is not used. This allows DDP to perform the allreduce operation, synchronizing the accumulated gradients across all processes before the optimizer.step().
- **`Optimizer Step and Zeroing`**: The `optimizer.step()` updates the model's parameters using the aggregated gradients, and `optimizer.zero_grad()` clears the gradients for the next accumulation cycle.

### Important Considerations

- **Loss Scaling (Mixed Precision)**: If using mixed-precision training with `torch.cuda.amp.GradScaler`, ensure proper scaling of the loss before `backward()` and unscaling before `optimizer.step()` when using gradient accumulation.
- **Learning Rate Adjustment**: When simulating a larger batch size, you might need to adjust the learning rate accordingly, often by scaling it up to maintain similar convergence properties.
- **Effective Batch Size**: The effective global batch size with gradient accumulation in DDP will be `num_gpus * mini_batch_size * accumulation_steps`.

---

## FSDP gradient accumulation

- Gradient accumulation in `FSDP1` & `FSDP2` is different. `FSDP1` uses `model.no_sync()` for not synchronizing gradients during intermediate steps, while `FSDP2` relies on `set_requires_gradient_sync()`.

### FSDP1

```python
# Important: Clear gradients before the loop starts
optimizer.zero_grad()

for i, (data, target) in enumerate(loader):
    # Determine if this is the final step in the accumulation window
    is_final_step = (i + 1) % accum_steps == 0

    if not is_final_step:
        # Intermediate steps: don't sync gradients
        with model.no_sync():
            out = model(data)
            loss = criterion(out, target) / accum_steps
            loss.backward()
    else:
        # Final step: gradients will be synced and we will step
        out = model(data)
        loss = criterion(out, target) / accum_steps
        loss.backward()
        
        # Now, update the model's weights
        optimizer.step()
        optimizer.zero_grad()
```

### FSDP2

```python
accumulation_steps = 4

for step, (data, target) in enumerate(dataloader):
    is_sync_step = ((step + 1) % accumulation_steps == 0)

    # Disable gradient sync for intermediate accumulation steps
    model.set_requires_gradient_sync(is_sync_step)

    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if is_sync_step:
        optimizer.step()
        optimizer.zero_grad()
```
