---
title: Sample lightning code
type: docs
prev: docs/01-basics/03-lightning
prev: docs/01-basics/03-lightning/02-pytorch-lightning.md
sidebar:
  open: false
weight: 22
---

## Sample `pytorch lightning` code

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L 
from lightning.pytorch.loggers import CSVLogger

# Dummy dataset
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16)

# Simple model
class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)  # Logged to CSV
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# CSV logger
csv_logger = CSVLogger(save_dir="logs", name="my_model", sub_dir="alisha")

# Trainer with CSV logger
trainer = L.Trainer(
    max_epochs=5,
    logger=csv_logger,
    log_every_n_steps=1
)

# Train
model = SimpleModel()
trainer.fit(model, loader)
```

## Sample `lightning fabric` code

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger

# Dummy dataset
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=16)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)

# Training loop
def train_one_epoch(fabric, model, optimizer, dataloader, loss_fn, epoch):
    model.train()
    for step, (x, y) in enumerate(dataloader):
        x, y = fabric.to_device((x, y))
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        fabric.backward(loss)
        optimizer.step()

        # log step loss
        fabric.log("train_loss", loss.item(), step=epoch * len(dataloader) + step)
        print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

# Fabric + logger
logger = CSVLogger(root_dir="logs", name="fabric_model", sub_dir="biwi_meri_pyari_si")
fabric = Fabric(loggers=logger)
fabric.launch()

# Setup
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
model, optimizer, loader = fabric.setup(model, optimizer, loader)

# Run training
for epoch in range(5):
    train_one_epoch(fabric, model, optimizer, loader, loss_fn, epoch)

# Force logger to write the CSV
logger.finalize("success")
```
