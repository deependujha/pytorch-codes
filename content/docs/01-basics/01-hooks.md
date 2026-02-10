---
title: Hooks, Backward Pass & retain_graph
type: docs
sidebar:
  open: false
weight: 2
math: true
---

## Backward Pass

### Forward vs backward computation

Forward pass is just value construction:

\[
\begin{align*}
y_1 &= f(x, W_1) \\
y_2 &= f(y_1, W_2) \\
y_3 &= f(y_2, W_3) \\
L &= \text{LossFn}(y_3, t)
\end{align*}
\]

Nothing “learns” here. We’re just building a computation graph.

Backward pass is **graph traversal in reverse**, applying the chain rule locally at each node.

Calling:

```python
loss.backward()
```

means:

\[
\frac{\partial L}{\partial L} = 1
\]

This is not deep, just the seed. From here, every node asks:

> “Given the gradient of my output, how do I produce gradients of my inputs and parameters?”

---

### Step-by-step backward flow

1. **Loss node**

\[
\frac{\partial L}{\partial y_3}
\quad\text{(defined by the loss function)}
\]

2. **Layer 3**

Given ( $y_3 = f(y_2, W_3)$ ):

\[
\begin{align*}
\text{gradients w.r.t. weights} &= \frac{\partial L}{\partial W_3} &= \frac{\partial L}{\partial y_3}
\cdot \frac{\partial y_3}{\partial W_3}
\\
\\
\text{gradients w.r.t. input} &= \frac{\partial L}{\partial y_2} &= \frac{\partial L}{\partial y_3}
\cdot
\frac{\partial y_3}{\partial y_2}
\end{align*}
\]

Two outputs. Always two. One gradient is for updating parameters and learning, and the other is for propagating the learning signal backward to earlier layers.

3. **Layer 2**

Same pattern:

\[
\begin{align*}
\text{gradients w.r.t. weights} &= \frac{\partial L}{\partial W_2} &= \frac{\partial L}{\partial y_2}
\cdot \frac{\partial y_2}{\partial W_2}
\\
\\
\text{gradients w.r.t. input} &= \frac{\partial L}{\partial y_1} &= \frac{\partial L}{\partial y_2}
\cdot
\frac{\partial y_2}{\partial y_1}
\end{align*}
\]

And so on until the input.

This is the whole backward pass.

---

### Why we compute **both** gradients

Every layer consumes a gradient and produces **two kinds of gradients**:

```text
incoming gradient
        ↓
┌─────────────────┐
│     Layer       │
└─────────────────┘
   ↓           ↓
parameter grad   input grad
```

* **Parameter gradients** are used by the optimizer to update the model’s weights.
* **Input gradients** propagate the learning signal backward so earlier layers can compute their own gradients.

If we only computed parameter gradients, the learning signal would stop at that layer. Earlier layers would receive nothing, and learning would silently fail. Backward is therefore not just about updating weights; it is about **moving information backward through the graph**.

Concretely: to compute gradients for **Layer 1**, we must receive the input gradient produced by **Layer 2**. If that input gradient is missing, the chain rule breaks.

---

### Where the dependency shows up (explicitly)

Let:

\[
h = \text{Layer}_1(x), \quad y = \text{Layer}_2(h), \quad L = \text{Loss}(y)
\]

Then the gradient of the loss w.r.t. Layer 1’s parameters is:

\[
\frac{\partial L}{\partial W_1} =
\underbrace{\frac{\partial L}{\partial h}}_{\text{from Layer 2}}
\cdot
\frac{\partial h}{\partial W_1}
\]

And that term:

\[
\frac{\partial L}{\partial h} =
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial h}
\]

is **exactly the input gradient computed by Layer 2**.

No input gradient from Layer 2 → no \( \frac{\partial L}{\partial h} \) → no \( \frac{\partial L}{\partial W_1} \).

> No input gradient → chain rule breaks → learning stops.

> [!NOTE]
> $\frac{\partial h}{\partial W_1}$ comes from Layer 1’s own forward definition.
> $(h = f(x, W_1))$ and is a local derivative independent of the loss or Layer 2.

* upstream gradient → from next layer
* local Jacobian → from *this* layer’s forward math

This is why backward is not “just about weights”. It’s about **signal propagation**.

---

## `retain_graph=True`

During the forward pass, PyTorch builds a computation graph that records all operations involving tensors with `requires_grad=True`. This graph is later traversed during the backward pass to compute gradients via the chain rule.

By default, **calling `backward()` frees the computation graph** to save memory. Once freed, the same graph cannot be used again.

```python
import torch
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2

y.backward()   # computes dy/dx and frees the graph
y.backward()   # RuntimeError: graph has been freed
```

This behavior is intentional: most training loops only need **one backward pass per forward pass**.

---

### What `retain_graph=True` changes

Setting:

```python
y.backward(retain_graph=True)
```

tells autograd **not to free the computation graph** after the backward pass.

This allows you to call `backward()` multiple times on the **same forward graph**, at the cost of higher memory usage.

```python
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2

y.backward(retain_graph=True)
print(x.grad)   # 2.0

y.backward()
print(x.grad)   # 4.0 (gradients accumulate)
```

Important detail: gradients **accumulate** by default. The second backward adds to the first.

> [!WARNING]
> Ensure last backward call has `retain_graph=False` (or omitted) to free memory after all necessary backward passes are done.

---

### When `retain_graph=True` is actually needed

You need it only when **multiple backward passes depend on the same forward computation**.

**1. Multiple losses from the same graph:**

- e.g., adversarial setups where multiple losses depend on overlapping parts of the same forward graph

```python
y = model(x)
loss1 = L1(y)
loss2 = L2(y)

loss1.backward(retain_graph=True)
loss2.backward()
```

Here, both `loss1` and `loss2` depend on the same `y`, so the graph must survive the first backward.

On the **final backward**, `retain_graph` should be `False` (or omitted) so the graph can be freed and memory released.

**2. Higher-order gradients**

When computing gradients of gradients:

```python
g = torch.autograd.grad(loss, w, create_graph=True)
g.backward()
```

The first gradient builds a new graph; the second backward needs the original structure intact.

---

## Hooks in PyTorch tensors and modules

Hooks allow you to **inject custom behavior** at specific points in the forward or backward pass without modifying the underlying code of the model or loss function.

- when you attach a hook to a tensor or module, it will be called automatically during the forward or backward pass at the appropriate time.

- attaching hook returns a `handle` that can be used to remove the hook later to prevent it from being called in future passes and free up resources.

- **`Hooks observe or modify values at graph execution time, not graph construction time.`**

### Get all register hooks

```python
import torch
import torch.nn as nn


def is_register_hook(name: str):
    return name.startswith("register") and name.endswith("hook")


def get_all_register_hooks(obj: object):
    return [fn for fn in dir(obj) if is_register_hook(fn)]

def main():
    # tensor register hooks
    x = torch.tensor(1.0, requires_grad=True)
    print(f"Tensor Register Hooks: {get_all_register_hooks(x)}")

    print("\n")

    # module register hooks
    linear_module = nn.Linear(2, 2)
    print(f"Module Register Hooks: {get_all_register_hooks(linear_module)}")


if __name__ == "__main__":
    main()
```

<details>
<summary>Output</summary>

```md
Tensor Register Hooks: ['register_hook', 'register_post_accumulate_grad_hook']


Module Register Hooks: ['register_backward_hook', 'register_forward_hook',
 'register_forward_pre_hook', 'register_full_backward_hook',
 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook',
 'register_load_state_dict_pre_hook', 'register_state_dict_post_hook',
 'register_state_dict_pre_hook']
```
</details>

### Tensor Hooks 🪝

#### `register_hook`

- allows you to register a hook on a tensor that will be called every time a gradient is computed for that tensor during the backward pass. It doesn't care if the gradient is accumulated or not. It will be called for every backward pass.

#### `register_post_accumulate_grad_hook`

- is a more specific hook that is called **after** the gradient for the tensor has been accumulated. This means it will only be called once per backward pass, after all contributions to the gradient have been summed up. It gives the actual `gradient value` stored in the `.grad` attribute of the tensor.

```python
import torch

w = torch.nn.Parameter(torch.tensor(1.0))

def hook1(grad):
    print("register_hook:", grad)

def hook2(p):
    print("post_accumulate:", p.grad)

handle_1 = w.register_hook(hook1)
handle_2 = w.register_post_accumulate_grad_hook(hook2)

y1 = w * 2 # w gets a gradient of 2

# retain_graph=True is needed to allow multiple backward calls on the same graph
y1.backward(retain_graph=True)

y2 = w * 3 # w gets an additional gradient of 3, total should be 5
y2.backward()

# cleanup
handle_1.remove()
handle_2.remove()
```

<details open>
<summary>Output</summary>

```md
register_hook: tensor(2.)
post_accumulate: tensor(2.)
register_hook: tensor(3.)
post_accumulate: tensor(5.)
```

- both hooks are called during each backward pass, but `register_hook` receives the gradient contribution from that specific backward, while `register_post_accumulate_grad_hook` receives the total accumulated gradient after all contributions have been summed.

- most of the time, you want `register_post_accumulate_grad_hook` to get the final gradient value. Use `register_hook` if you need to see or modify the gradient contribution from each individual backward pass before accumulation.
</details>

---


### Module Hooks 🪝

#### `register_forward_pre_hook`

- is called **before** the forward method of the module is executed. It allows you to inspect or modify the input to the module before it processes it.

```python
my_module = torch.nn.Linear(2, 2)
input_tensor = torch.tensor([[1.0, 2.0]])

def hook_fn(_module, _input):
    print(f"Inside forward pre-hook")
    print(f"module: {_module.__class__.__name__}")
    print(f"input: {_input}")

handle = my_module.register_forward_pre_hook(hook_fn)
output = my_module(input_tensor)
handle.remove()  # Remove the hook when done
```

#### `register_forward_hook`

- is called **after** the forward method of the module has executed. It allows you to inspect or modify the output of the module after it has processed the input.

```python
my_module = torch.nn.Linear(2, 2)
input_tensor = torch.tensor([[1.0, 2.0]])

def hook_fn(_module, _input, _output):
    print(f"Inside forward hook")
    print(f"module: {_module.__class__.__name__}")
    print(f"input: {_input}")
    print(f"output: {_output}")

handle = my_module.register_forward_hook(hook_fn)
output = my_module(input_tensor)
handle.remove()  # Remove the hook when done
```

#### `register_full_backward_hook`

- is called during the backward pass, after the gradients have been computed for the module. It allows you to inspect or modify the gradients with respect to the inputs and parameters of the module.

```python
my_module = torch.nn.Linear(2, 2)
input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)

def hook_fn(_module, grad_input, grad_output):
    print(f"Inside full backward hook")
    print(f"module: {_module.__class__.__name__}")
    print(f"grad_input: {grad_input}")
    print(f"grad_output: {grad_output}")

handle = my_module.register_full_backward_hook(hook_fn)
output = my_module(input_tensor)
output.backward()
handle.remove()  # Remove the hook when done
```

#### `register_full_backward_pre_hook`

- is called during the backward pass, **after** the gradients are computed for the `module`, but **before** the gradients are computed for `input`.

```python
my_module = torch.nn.Linear(2, 2)
input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)

def hook_fn(_module, grad_output):
    print(f"Inside full backward pre-hook")
    print(f"module: {_module.__class__.__name__}")
    print(f"grad_output: {grad_output}")

handle = my_module.register_full_backward_pre_hook(hook_fn)
output = my_module(input_tensor)
output.backward()
handle.remove()  # Remove the hook when done
```

#### State dict hooks

- `register_state_dict_pre_hook` and `register_state_dict_post_hook` allow you to modify the state dictionary of a module before and after saving/loading.

```python
import torch

my_module = torch.nn.Sequential(
    torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
)


# can be used to perform pre-processing before the `state_dict` call is made.
def pre_hook_fn(_module, state_dict, keep_vars):
    # state_dict is not actually a dict here, as call is yet to be made.
    print()
    print(f" Inside state dict pre-hook ".center(50, "-"))
    print(f"module: {_module.__class__.__name__}")
    print(f"state_dict keys: {type(state_dict)}")  # str
    print(f"keep_vars: {keep_vars}")
    print()


# can modify the state_dict inplace
def post_hook_fn(_module, state_dict, prefix, local_metadata):
    print()
    print(f" Inside state dict post-hook ".center(50, "-"))
    print(f"module: {_module.__class__.__name__}")
    print(f"state_dict keys: {list(state_dict.keys())}")
    print(f"prefix: {prefix}")
    print(f"local_metadata: {local_metadata}")
    print()


pre_handle = my_module.register_state_dict_pre_hook(pre_hook_fn)
post_handle = my_module.register_state_dict_post_hook(post_hook_fn)
torch.save(my_module.state_dict(), "model_state.pth")

# cleanup: remove hook handles
pre_handle.remove()
post_handle.remove()
```

#### loading state dict hooks

- `register_load_state_dict_pre_hook` and `register_load_state_dict_post_hook` allow you to modify the state dictionary of a module before and after loading.

```python
import torch

my_module = torch.nn.Sequential(
    torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
)

def pre_hook_fn(_module, state_dict, *args):
    # hook(module, state_dict, prefix, local_metadata, strict, missing_keys,
    # unexpected_keys, error_msgs) -> None 
    print()
    print(f" Inside load state dict pre-hook ".center(50, "-"))
    print(f"module: {_module.__class__.__name__}")
    print(f"state_dict keys: {list(state_dict.keys())}")
    print(f"args: {args}")
    print()

def post_hook_fn(_module, incompatible_keys):
    print()
    print(f" Inside load state dict post-hook ".center(50, "-"))
    print(f"module: {_module.__class__.__name__}")
    print(f"incompatible keys: {incompatible_keys}")  # str

pre_handle = my_module.register_load_state_dict_pre_hook(pre_hook_fn)
post_handle = my_module.register_load_state_dict_post_hook(post_hook_fn)
my_module.load_state_dict(torch.load("model_state.pth"))

# cleanup: remove hook handles
pre_handle.remove()
post_handle.remove()
```
