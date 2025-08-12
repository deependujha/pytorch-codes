---
title: Transfer Learning
type: docs
prev: docs/01-basics/02-concepts/03-gradient-clipping.md
sidebar:
  open: false
weight: 24
---


Transfer learning uses a pretrained model’s knowledge (features, weights) for a new but related task, instead of training from scratch.

## Why Use It?

* Faster convergence (less data needed).
* Better performance when dataset is small.
* Leverages features learned from large datasets (e.g., ImageNet).

---

## Two Main Strategies

### **1. Feature Extraction**

* Freeze backbone weights → only train new head.
* Keeps pretrained features intact.

```python
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
```

### **2. Fine-Tuning**

* Unfreeze some/all layers → train with a low LR.
* Allows adaptation of pretrained features to new task.

```python
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

## Common Operations on `nn.Module`

**Replace Last Layer**

```python
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)    # ResNet example
model.classifier[6] = torch.nn.Linear(4096, num_classes)          # VGG example
```

**Remove a Layer**

```python
from torch import nn
model.fc = nn.Identity()   # Acts like a no-op
```

**Slice Sequential Models**

- If your model is `nn.Sequential`, you can cut it easily:

```python
features = nn.Sequential(*list(model.children())[:-1])
```

**Inspect Model Parts**

- If your model is complex, and to replace last layer, we need to know the name of the last layer.

```python
for name, module in model.named_children():
    print(name, module)
```

> From the output, you can decide which layers to replace. Generally, you would replace the last layer with a new one that has the correct output size for your specific task.

---

## Freezing & Unfreezing Layers

* **Freeze** → set `requires_grad=False` (no weight updates, no gradient memory).
* **Unfreeze** → set `requires_grad=True`.
* Can freeze entire model or specific submodules.

---

## `model.eval()` and `torch.no_grad()` in Transfer Learning

**`model.eval()`**

* Sets layers like **Dropout** and **BatchNorm** to inference mode (no randomness, uses stored stats).
* Does **not** disable gradient calculation.

**`torch.no_grad()`**

* Disables gradient tracking (saves memory, speeds up inference).
* Does **not** change layer behavior.

**Typical inference pattern:**

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

Use during:

* Validation
* Testing
* Feature extraction (when you’re not updating weights)

---

## Good Practices

* Lower LR when fine-tuning pretrained layers (`~1e-4` or smaller).
* Normalize inputs with same mean/std as the pretrained model.
* Freeze/unfreeze in stages for stable training.
* Always call `model.eval()` + `torch.no_grad()` for inference/feature extraction.
* Track LR and `requires_grad` status:

```python
[p.requires_grad for p in model.parameters()]
```

---

## Example: ResNet Feature Extraction

```python
import torchvision.models as models
resnet = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace classifier
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

# Inference
resnet.eval()
with torch.no_grad():
    features = resnet(inputs)
```
