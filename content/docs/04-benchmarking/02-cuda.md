---
title: CUDA Streams and Events
type: docs
sidebar:
  open: false
weight: 29
math: false
---

CUDA operations in PyTorch execute **asynchronously**. To manage ordering, parallelism, and dependencies between GPU operations, CUDA provides two core primitives:

* **Streams** → *where* operations are queued
* **Events** → *when* operations are considered complete

Understanding these is essential for correct synchronization and performance optimization.

### correct way to time cuda code

```python
torch.cuda.synchronize() # Blocks CPU until all streams on the device finish
t0 = time.time()

y = x @ W
z = torch.relu(y)

torch.cuda.synchronize() # Blocks CPU until all streams on the device finish
t1 = time.time()

print(t1 - t0)
```

---

## CUDA Streams

A **CUDA stream** is an **ordered queue of GPU operations**.

### Key properties

* Operations **within the same stream** execute **sequentially**
* Operations **across different streams** *may execute in parallel*
* All CUDA ops are asynchronous with respect to the CPU

By default, PyTorch uses the **default stream** for each device.

---

### Creating and using streams

```python
s = torch.cuda.Stream()

with torch.cuda.stream(s):
    y = x @ W
```

This enqueues `x @ W` into stream `s` instead of the default stream.

---

### Mental model

```
Stream = command queue
GPU executes commands from queues asynchronously
```

---

## CUDA Events

A **CUDA event** is a **completion marker** placed into a stream.

It represents the point at which:

> *All operations enqueued before the event are finished.*

Events are used to express **dependencies between streams** without globally synchronizing the device.

---

### Creating an event

```python
event = torch.cuda.Event()
```

This creates an empty event that is not yet associated with any stream.

---

### Recording an event

```python
event.record(s1)
```

This inserts the event into stream `s1` at the current position.

Meaning:

* The event is considered *complete* only after all previously enqueued work in `s1` finishes.

Visual timeline:

```
s1:  op1 → op2 → [EVENT] → op3
```

---

## Waiting on an event

```python
s2.wait_event(event)
```

This tells CUDA:

> “Pause execution of `s2` until the event is complete.”

Important:

* Only operations **after** `wait_event` are blocked
* Other streams continue executing
* This is **fine-grained synchronization**

---

## Example: stream dependency using events

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(s1):
    data = torch.randn(1000, 1000, device="cuda")

event.record(s1)

with torch.cuda.stream(s2):
    s2.wait_event(event)
    result = torch.sum(data)

s2.synchronize()
```

### What this guarantees

* `torch.sum(data)` runs **only after** `data` is fully initialized
* No global device synchronization
* Maximum possible overlap elsewhere

---

## Stream vs Event synchronization

### `stream.synchronize()`

```python
s2.synchronize()
```

* Blocks CPU until **all work in that stream** finishes
* Stream-specific barrier

---

### `torch.cuda.synchronize()`

```python
torch.cuda.synchronize()
```

* Blocks CPU until **all streams on the device** finish
* Global barrier (expensive)

---

### `event.synchronize()`

```python
event.synchronize()
```

* Blocks CPU until **that specific event** completes
* More precise than `torch.cuda.synchronize()`

---

## Why events exist (important)

Without events, the only safe synchronization option is:

```python
torch.cuda.synchronize()
```

Which:

* Kills overlap
* Reduces performance
* Forces global serialization

Events enable:

* Overlapping compute and communication
* Pipeline parallelism
* Asynchronous data transfers
* Precise correctness guarantees

---

## Streams vs Events (summary)

| Concept      | Purpose                         |
| ------------ | ------------------------------- |
| Stream       | Orders GPU work                 |
| Event        | Marks completion point          |
| `record`     | Places marker in a stream       |
| `wait_event` | Creates cross-stream dependency |
| Synchronize  | Blocks CPU until completion     |

---

## Common pitfalls

* Timing GPU code without synchronization → incorrect results
* Using `torch.cuda.synchronize()` instead of events → lost parallelism
* Assuming tensor creation implies readiness (it doesn’t)

---

## One-line mental model

> **Streams schedule work. Events signal completion. Synchronization enforces correctness.**

---

## Summary

* CUDA streams are ordered queues of GPU operations
* CUDA events mark completion points within streams
* Events allow fine-grained synchronization across streams
* Proper use avoids global synchronization and improves performance
* Streams provide concurrency; events provide correctness
