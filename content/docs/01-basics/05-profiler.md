---
title: PyTorch Profiler
type: docs
sidebar:
  open: false
weight: 500
---

PyTorch Profiler is a tool that allows the collection of performance metrics during training and inference. Profiler’s context manager API can be used to better understand what model operators are the most expensive, examine their input shapes and stack traces, study device kernel activity and visualize the execution trace.

- It can be used to profile both CPU and GPU operations, providing insights into the performance bottlenecks in your model.
- The profiler can be integrated with visualization tools like TensorBoard or Chrome Trace Viewer for detailed analyis.
- use `torch.profiler.profile` to profile the code block. It internally uses `_kinetoProfile` which is a low-level profiler wrapper around `torch.autograd.profiler`. Most of the time, you will use `torch.profiler.profile` to profile your code.

---

## Profile code block

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

# Print the profiling results
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
```

### Arguments:

- `activities`: List of activities to profile, such as CPU and CUDA.
- `profile_memory`: If set to `True`, memory usage will be tracked.
- `record_shapes`: If set to `True`, the shapes of the inputs and outputs will be recorded.
- `with_stack`: If set to `True`, stack traces will be recorded for each operation. Increases overhead but useful for debugging.
- `use_cuda`: **Deprecated**, If set to `True`, CUDA operations will be profiled. Use `ProfilerActivity.CUDA` instead by specifying it in the `activities` argument.
- `schedule`: callable that takes step (int) as a single parameter and returns `ProfilerAction` value that specifies the profiler action to perform at each step.
- `on_trace_ready`: callable that is called at each step when schedule returns `ProfilerAction.RECORD_AND_SAVE` during the profiling.

---

## PyTorch Profiler with `Schedule`

- Returns a callable that can be used as profiler `schedule` argument. 

- The profiler will skip the first `skip_first` steps,
- then wait for `wait` steps,
- then do the warmup for the next `warmup` steps,
- then do the active recording for the next `active` steps and then repeat the cycle starting with `wait` steps.
- The optional number of cycles is specified with the `repeat` parameter, the zero value means that the cycles will continue until the profiling is finished.
- The `skip_first_wait` parameter controls whether the first wait stage should be skipped. This can be useful if a user wants to wait longer than `skip_first` between cycles, but not for the first profile. For example, if `skip_first` is 10 and `wait` is 20, the first cycle will wait 10 + 20 = 30 steps before warmup if `skip_first_wait` is zero, but will wait only 10 steps if `skip_first_wait` is non-zero. All subsequent cycles will then wait 20 steps between the last active and warmup.

```python
# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    )
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=trace_handler,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
) as p:
    for iter in range(N):
        code_iteration_to_profile(iter)
        # send a signal to the profiler that the next iteration has started
        p.step()
```

- doing `p.step()` at the end of each iteration is necessary to signal the profiler that a new iteration has started. It internally calls `schedule(iter_count)` and if the result is `ProfilerAction.RECORD_AND_SAVE`, it will call `on_trace_ready` with the profiler object.

---

## `ProfilerAction`

- it is an enum that defines the actions that the profiler can take at each step of the training loop.

- [original code on github](https://github.com/pytorch/pytorch/blob/97bd4db004ef2527f3e251247a29d822e83a3d97/torch/profiler/profiler.py#L490-L498)

```python
class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3
```

---

## Save chrome's `trace.json`

```python
prof.export_chrome_trace("my_trace.json")
```

---

## PyTorch Profiler with `Tensorboard`

- Install PyTorch Profiler TensorBoard Plugin.

```bash
pip install torch_tb_profiler
```

- Use `torch.profiler.tensorboard_trace_handler` to save the profiling results in a format compatible with TensorBoard.
```python
# prof - PyTorch Profiler object
dir_name = "./log"  # Directory to save the trace files
torch.profiler.tensorboard_trace_handler(dir_name)(prof)
```

- Start TensorBoard to visualize the profiling results.

```bash
tensorboard --logdir=./log
```

---

## PyTorch Profiler with `Tensorboard` & `Schedule`

- pass `torch.profiler.tensorboard_trace_handler` to `on_trace_ready` argument.

```python
# without using `with` context manager
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True)

prof.start() # Start the profiler

for step, batch_data in enumerate(train_loader):
    prof.step() # Signal the profiler that a new step has started
    if step >= 1 + 1 + 3:
        break
    train(batch_data)

prof.stop() # Stop the profiler
```

- alternatively, you can use the `with` context manager to automatically start and stop the profiler.

```python
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        with_stack=True) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # Signal the profiler that a new step has started
        if step >= 1 + 1 + 3:
            break
        train(batch_data)
```

original code for enter and exit methods of the context manager:

```python
class profile(_KinetoProfile):
    ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()
```

---

## Switching between CPU and GPU profiling mid execution

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    code_to_profile_0()

    # turn off collection of all CUDA activity
    p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
    code_to_profile_1()

    # turn on collection of all CUDA activity
    p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
    code_to_profile_2()

print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))
```

---

## `PyTorch Lightning Profiler`

- PyTorch Lightning's `pytorch profiler` uses `ScheduleWrapper` to wrap the `pytorch.profiler.schedule`, as pytorch's schedule tracks only one action, but `lightning's schedulewrapper` can perform recording for `training`, `validation`, and `test` separately.

- [see lightning's pytorch profiler implementation](https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/profilers/profiler.py)
