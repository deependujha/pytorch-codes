---
title: PyTorch Profiler
type: docs
sidebar:
  open: false
weight: 500
math: true
---

PyTorch Profiler is a performance analysis tool for profiling **training and inference** workloads. It enables inspection of:

* CPU and GPU operator execution
* CUDA kernel launches and execution
* Memory allocation and usage
* Input shapes and call stacks
* Execution timelines (CPU–GPU overlap)

The profiler integrates with visualization tools such as **TensorBoard** and **Chrome Trace Viewer**.

Most users interact with the profiler via:

```python
torch.profiler.profile
```

Internally, this API is built on top of **Kineto**, via the internal `_KinetoProfile` class, which acts as a low-level wrapper around `autograd.profiler`.

---

## Basic profiling of a code block

```python
import torch

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
) as p:
    code_to_profile()

print(p.key_averages().table(
    sort_by="self_cuda_time_total",
    row_limit=-1,
))
```

### Common arguments

* **`activities`**
  Specifies which activities to record:

  * `ProfilerActivity.CPU`
  * `ProfilerActivity.CUDA`

* **`profile_memory`**
  Tracks memory allocation and deallocation events.

* **`record_shapes`**
  Records input/output tensor shapes for each operator.

* **`with_stack`**
  Records Python stack traces for each operator.
  ⚠️ Significantly increases overhead.

* **`use_cuda`**
  ❌ Deprecated. Use `ProfilerActivity.CUDA` instead.

* **`schedule`**
  Callable controlling when profiling is active.

* **`on_trace_ready`**
  Callback invoked when a trace becomes available (e.g., export to TensorBoard or Chrome trace).

---

## Profiling with a schedule

Profiling every iteration is expensive. Instead, PyTorch supports **scheduled profiling**, which divides execution into phases.

### Schedule phases

A schedule consists of:

1. **skip_first** – initial steps ignored entirely
2. **wait** – profiler inactive
3. **warmup** – profiler active but data discarded
4. **active** – profiler records data
5. **repeat** – number of cycles (0 = infinite)

This reduces overhead and captures representative iterations.

---

### Example with schedule

```python
def trace_handler(prof):
    print(
        prof.key_averages()
        .table(sort_by="self_cuda_time_total", row_limit=-1)
    )
    # prof.export_chrome_trace(
    #     f"/tmp/trace_{prof.step_num}.json"
    # )

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1,
    ),
    on_trace_ready=trace_handler,
) as p:
    for it in range(N):
        code_iteration_to_profile(it)
        p.step()
```

### Important note on `p.step()`

* `p.step()` **must be called once per iteration**
* It advances the internal step counter
* Internally:

  * Calls `schedule(step)`
  * Triggers `on_trace_ready` when `RECORD_AND_SAVE`

---

## `ProfilerAction`

`ProfilerAction` defines what the profiler does at each step.

```python
class ProfilerAction(Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
```

Meaning:

* **NONE** → profiler disabled
* **WARMUP** → collect but discard data
* **RECORD** → collect data
* **RECORD_AND_SAVE** → collect and trigger callback

---

## Exporting traces

### Chrome Trace

```python
prof.export_chrome_trace("trace.json")
```

* Open in `chrome://tracing`
* Best for low-level CUDA + timeline inspection

---

## Profiling with TensorBoard

### Installation

```bash
pip install torch_tb_profiler
```

### Writing traces

```python
torch.profiler.tensorboard_trace_handler("./log")(prof)
```

### Launch TensorBoard

```bash
tensorboard --logdir=./log
```

TensorBoard provides:

* Operator breakdowns
* Kernel timelines
* Memory views
* Overlap visualization

---

## TensorBoard profiling with schedule

### Without context manager

```python
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./log/resnet18"
    ),
    record_shapes=True,
    with_stack=True,
)

prof.start()

for step, batch in enumerate(train_loader):
    prof.step()
    if step >= 1 + 1 + 3:
        break
    train(batch)

prof.stop()
```

---

### With context manager (recommended)

```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1,
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./log/resnet18"
    ),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        prof.step()
        if step >= 1 + 1 + 3:
            break
        train(batch)
```

---

### Context manager internals (for reference)

```python
class profile(_KinetoProfile):
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        prof.KinetoStepTracker.erase_step_count(
            PROFILER_STEP_NAME
        )
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()
```

---

## Dynamically enabling / disabling activities

Profiling can be toggled **mid-execution** without restarting the profiler.

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    code_to_profile_0()

    p.toggle_collection_dynamic(
        False, [torch.profiler.ProfilerActivity.CUDA]
    )
    code_to_profile_1()

    p.toggle_collection_dynamic(
        True, [torch.profiler.ProfilerActivity.CUDA]
    )
    code_to_profile_2()

print(
    p.key_averages()
    .table(sort_by="self_cuda_time_total", row_limit=-1)
)
```

---

## PyTorch Lightning Profiler

* PyTorch Lightning wraps `torch.profiler` using `ScheduleWrapper`
* This allows **separate profiling** of:

  * training
  * validation
  * testing
* Necessary because vanilla PyTorch schedules track a single action stream

Reference implementation:

* [https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/profilers/profiler.py](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/profilers/profiler.py)

---

## Important caveats (worth remembering)

* Always synchronize before timing manually:

  ```python
  torch.cuda.synchronize()
  ```
* `.item()`, `.cpu()`, and printing CUDA tensors cause **implicit synchronization**
* Profiling adds overhead — never profile the entire training loop
* Prefer **few active steps** with warmup
* High `self_cuda_time` usually indicates kernel cost
* High CPU time with low CUDA utilization often indicates synchronization or launch overhead

---

## Summary

* PyTorch Profiler is built on **Kineto**
* Supports CPU, CUDA, memory, and timeline profiling
* Scheduling is essential to reduce overhead
* `p.step()` drives the profiler state machine
* TensorBoard and Chrome Trace provide complementary views
* Profiler is a diagnostic tool — not a benchmark by itself
