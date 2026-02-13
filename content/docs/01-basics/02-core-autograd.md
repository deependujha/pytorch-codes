---
title: Core Autograd
type: docs
sidebar:
  open: false
weight: 3
math: true
---

```python
def print_tensor_info(tensor):
    print(
        f"tensor: {tensor}, requires_grad: {tensor.requires_grad}, "
        "grad_fn: {tensor.grad_fn}, grad: {tensor.grad}, is_leaf: {tensor.is_leaf}"
    )
```

- a tensor that `requires_grad` will track all operations on it, and the gradient will be calculated during backpropagation. To track gradients, `autograd` uses `grad_fn` to record the function that created the tensor. If a tensor is created by the user and not as a result of an operation, it is called a leaf tensor and has `is_leaf=True`. The gradient of a leaf tensor will be stored in the `grad` attribute after backpropagation.
For intermediate tensors, the `grad` attribute will be `None` because they are not leaf tensors, and their gradients are not stored. However, they still have a `grad_fn` that points to the function that created them, allowing autograd to compute gradients for these tensors during backpropagation.

- a tensor created by some operation on tensors that have `requires_grad=False` has `requires_grad=False` and `grad_fn=None` and `is_leaf=True`.

```python
a = torch.tensor(1.0, requires_grad=False)
b = torch.tensor(2.0, requires_grad=False)

c = a * b

assert a.requires_grad == False and a.grad_fn is None and a.is_leaf == True
assert b.requires_grad == False and b.grad_fn is None and b.is_leaf == True
assert c.requires_grad == False and c.grad_fn is None and c.is_leaf == True
```

- a tensor created by some operation on tensors that has any tensor with `requires_grad=True` has `requires_grad=True` and `grad_fn` is not `None` (function used to create the tensor) and `is_leaf=False`.

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=False)

c = a * b

assert a.requires_grad == True and a.grad_fn is None and a.is_leaf == True
assert b.requires_grad == False and b.grad_fn is None and b.is_leaf == True
assert c.requires_grad == True and c.grad_fn is not None and c.is_leaf == False
```

---

## `grad_fn`

![grad_fn](/autograd/autograd.png)

- `grad_fn` is an attribute of a tensor that points to the function that created the tensor. It is used by autograd to track the operations performed on tensors and to compute gradients during backpropagation. If a tensor is created by the user and not as a result of an operation, it will have `grad_fn=None`. If a tensor is created by an operation on tensors that have `requires_grad=True`, it will have a non-`None` `grad_fn` that points to the function used to create it.

- `grad_fn.next_functions` is a tuple that contains the next functions in the computational graph. Each element in the tuple is a pair of (function, index), where `function` is the function that created the tensor and `index` is the index of the input tensor to that function. This allows autograd to traverse the computational graph and compute gradients for all tensors involved in the operations.

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=False)

c = a + b
d = a * b
e = a / b
f = a - b
g = a ** b
h = torch.sin(a)
i = torch.cos(a)
k = torch.exp(a)
l = torch.log(a)

def print_grad_fn_next_functions(x, tensor):
    _str = f"print_grad_fn_next_functions({x}) # "
    _z = tensor.grad_fn.__class__.__name__
    out = _str + f"{_z}: => {tensor.grad_fn.next_functions}"
    print(out)  

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=False)

c = a + b
d = a * b
e = a / b
f = a - b
g = a ** b
h = torch.sin(a)
i = torch.cos(a)
k = torch.exp(a)
l = torch.log(a)

def print_grad_fn_next_functions(x, tensor):
    _z = tensor.grad_fn.__class__.__name__
    out = f"{_z}: => {tensor.grad_fn.next_functions}"
    print(out)  

# AddBackward0: => ((<AccumulateGrad object at 0x103babfd0>, 0), (None, 0))
print_grad_fn_next_functions(c)

# MulBackward0: => ((<AccumulateGrad object at 0x103babf70>, 0), (None, 0))
print_grad_fn_next_functions(d)

# DivBackward0: => ((<AccumulateGrad object at 0x103babfd0>, 0), (None, 0))
print_grad_fn_next_functions(e)

# SubBackward0: => ((<AccumulateGrad object at 0x103babf70>, 0), (None, 0))
print_grad_fn_next_functions(f)

# PowBackward1: => ((<AccumulateGrad object at 0x103babfd0>, 0), (None, 0))
print_grad_fn_next_functions(g)

# SinBackward0: => ((<AccumulateGrad object at 0x103babf70>, 0),)
print_grad_fn_next_functions(h)

# CosBackward0: => ((<AccumulateGrad object at 0x103babfd0>, 0),)
print_grad_fn_next_functions(i)

# ExpBackward0: => ((<AccumulateGrad object at 0x103babf70>, 0),)
print_grad_fn_next_functions(k)

# LogBackward0: => ((<AccumulateGrad object at 0x103babfd0>, 0),)
print_grad_fn_next_functions(l)
```

- first run `grad_fn` to get the gradient of the current function, and then run `next_functions` to get the gradient of the next functions (previous according to forward graph, and next according to backward graph).

- A tensor that doesn't requires gradient will be present in the `next_functions` as `None` because it doesn't have a `grad_fn` and doesn't require gradient.

- A tensor that requires gradient will be present in the `next_functions` as an `AccumulateGrad` object because it is a leaf tensor that requires gradient and its gradient will be accumulated (summed) during backpropagation.

- we can also apply hooks to the `grad_fn` of a tensor.

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.sin(a)
c = b * 2
def hook_fn(*grad_inputs):
    print(f"tuple of gradients w.r.t. that node’s inputs: {grad_inputs}")

b.grad_fn.next_functions[0][0].register_hook(hook_fn)
c.backward()
```

---

## what is `index` in `next_functions`?

- The `index` in `next_functions` is the index of the input tensor to the function that created the current tensor. It indicates which input tensor was used in the operation that created the current tensor. This is important for autograd to correctly compute gradients during backpropagation, as it needs to know which input tensors were involved in the operations that led to the creation of the current tensor.

- use `.unbind()` to split the tensor into its individual elements and then when a graph is created, the `index` in `next_functions` will correspond to the position of the input tensor in the original tensor before it was unbound.

```python
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b, c, d = a.unbind()

# tensor(1., grad_fn=<UnbindBackward0>) tensor(2., grad_fn=<UnbindBackward0>) tensor(3., grad_fn=<UnbindBackward0>)
print(b, c, d)


m = b * c
n = m * d

print(m.grad_fn.next_functions)  # ((<UnbindBackward0 object at 0x1038d2740>>, 0), (<UnbindBackward0 object at 0x1038d2740>>, 1))
print(n.grad_fn.next_functions)  # ((<UnbindBackward0 object at 0x1038d2740>>, 0), (<UnbindBackward0 object at 0x1038d2740>>, 2))
```

---

## Distributed training techniques and autograd

- many distributed training techniques use autograd intelligently to implement parallel solutions.

- During `backward()`, `grad` value is not updated for intermediate tensors. They already contain the `grad_fn.next_functions` which facilitates the gradient flow till the leaf tensor and `grad` value is updated for leaf tensors only.

- if you explicitly want to update the `grad` value for intermediate tensors, you can use `retain_grad()` method on the tensor before calling `backward()`. This will allow you to access the `grad` value for intermediate tensors after backpropagation.

```python
a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

c = a * b
c.retain_grad()  # Retain the gradient for the intermediate tensor
d = c*2

d.backward()
print(a.grad, b.grad, c.grad)  # Output: 4, 2, 2
```

- since, `grad` value is only updated for leaf tensors and they don't have `grad_fn`, and so we can't apply hooks to them for custom gradient manipulation, like, accumulating gradient in `main gradient`, adding post-backward callback to wait for gradient synchronization in distributed setting, marking parameter ready for synchronization, etc.

- To achieve this, we use the smart trick of creating a dummy tensor that has a grad_fn and then we can apply hooks to that dummy tensor's grad_fn to manipulate the gradient of the leaf tensor.

```python
a = torch.tensor(1.0, requires_grad=True)
dummy = a.reshape_as(a) # Create a dummy tensor that has a grad_fn

b = dummy * 2

def hook_fn(*grad_inputs):
    print(f"tuple of gradients w.r.t. that node’s inputs: {grad_inputs}")

main_node_grad_acc_fn = dummy.grad_fn.next_functions[0][0]
main_node_grad_acc_fn.register_hook(hook_fn)

b.backward()
```

---

## `.grad` vs `.main_grad`

for `gradient accumulation` and proper synchronization in distributed training, we can use `.main_grad` to store the accumulated gradient for a leaf tensor instead of using `.grad`. This allows us to have better control over the gradient accumulation process and ensures that the gradients are properly synchronized across different processes in a distributed training setting.

- we update the `.grad` to synchronized gradient value after the synchronization is done, and we can use `.main_grad` to store the accumulated gradient value before synchronization.

```python
# code from: https://github.com/huggingface/picotron/blob/0035cce0e04afd6192763b11efe50010d8ad0f71/picotron/data_parallel/data_parallel.py
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.
```

---

## `tensor.data` vs `tensor.detach()`

- `tensor.data` is a deprecated way to get a new tensor that shares the same storage as the original tensor but does not require gradients. It can lead to unexpected behavior and is not recommended for use.

- use, `tensor.detach()` to get a new tensor that shares the same storage as the original tensor but does not require gradients. This is the recommended way to get a tensor that does not require gradients while still sharing the same data.

```python
a = torch.tensor(1.0, requires_grad=True)
b = a.data  # Deprecated, not recommended
c = a.detach()  # Recommended way to get a tensor that does not require gradients
```

- using `tensor.detach()` is safer bcoz it explicitly stops gradient history, and keeps autograd's internal safety tracking alive. `tensor._version` is a counter that tracks the number of in-place operations performed on the tensor. When you use `tensor.detach()`, it creates a new tensor that shares the same data but does not require gradients, and it also keeps the version counter intact. This allows autograd to track changes to the original tensor and ensures that any in-place operations on the original tensor will not affect the detached tensor, thus maintaining the integrity of the computational graph. On the other hand, using `tensor.data` can lead to unexpected behavior because it does not update the version counter, which can cause issues with autograd's tracking of operations and may lead to incorrect gradient calculations.

- `tensor.data` bypasses autograd metadata, version tracking and safety checks.

---

## `torch.autograd's Variable`

- `torch.autograd.Variable` is the core C++ implementation of autograd in PyTorch. It is used in advanced usecases, where you want to have more control over the autograd mechanics.

### adding callback

- we can add a callback to be executed after the backward pass is done using `torch.autograd.Variable._execution_engine.queue_callback()`.
- This allows us to perform operations that need to be executed after the backward pass, such as synchronizing gradients in a distributed training setting.

```python
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)

    def forward(self, input):
        return self.l1(input)

def blitz():
    print("keep working")

def hook_fn(grad):
    Variable._execution_engine.queue_callback(blitz)
    print(f"{grad=})
    return grad

if __name__ == "__main__":
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    x = torch.tensor([3.0])
    out = model(x)
    out.register_hook(hook_fn)
    print(f"{out=}")
    out.sum().backward()
    print("going for opt.step()")
    opt.step()
    print("all done")
```

### Directly call `C++` autograd engine

- Pytorch's `backward` checks that the output and grad have the same shape, while `C++'s 'backward' does not`.
- trick used in `Megatron-LM's Pipeline parallelism`, as output tensor is already sent to the next stage, and the output tensor is only useful for its `.grad_fn` field and not its `.data` field, so we can directly call `C++'s 'backward'` without worrying about the shape of the output tensor and grad tensor.

- code taken from: https://github.com/NVIDIA/Megatron-LM/blob/e33c8f78a35765d5aa37475a144da60e8a2349d1/megatron/core/pipeline_parallel/schedules.py#L121

```python
    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
```

