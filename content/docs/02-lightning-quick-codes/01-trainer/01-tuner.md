---
title: Tuner
type: docs
prev: docs/01-basics/01-data
next: docs/01-basics/02-concepts
sidebar:
  open: false
weight: 12
---

- `Tuner` is a utility in PyTorch Lightning that helps you `find the optimal batch size` and `learning rate` for your model training.

- Here's an example of how to use the `Tuner` class:

```python
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner

trainer = Trainer()
tuner = Tuner(trainer)

# Find the optimal learning rate
lr = tuner.lr_find(model, dataloaders)

# Find the optimal batch size
batch_size = tuner.scale_batch_size(model, dataloaders)
```

---

## `Batch Size Finder`

```python
import lightning.pytorch as pl
import torch
import torch.utils.data
from typing import Any

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, STEP_OUTPUT
from lightning.pytorch.tuner import Tuner


class ToyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 10)

    def forward(self, toy_variable: torch.Tensor) -> Any:
        return self.layer(toy_variable)

    def training_step(self, batch: torch.Tensor) -> STEP_OUTPUT:
        return self.layer(batch).sum()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(params=self.parameters())


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, throw_error: bool):
        super().__init__()
        self.throw_error = throw_error

    def __len__(self):
        return 100

    def __getitem__(self, item):
        if self.throw_error:
            raise RuntimeError("CUDA error: out of memory")
        else:
            return torch.randn((5,))


class ToyDatamodule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 1

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(ToyDataset(throw_error=(self.batch_size >= 16)))


def main():
    trainer = pl.Trainer()
    tuner = Tuner(trainer=trainer)

    model: pl.LightningModule = ToyModel()
    toy_datamodule: pl.LightningDataModule = ToyDatamodule()

    batch_size: int = tuner.scale_batch_size(
        model=model,
        datamodule=toy_datamodule,
        method="fit",
        steps_per_trial=3,
        max_trials=25,
    )
    print(f"This batch_size is {batch_size}.")


if __name__ == '__main__':
    main()
```

---

## `Learning Rate Finder`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
import lightning as L
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner.tuning import Tuner


class MinimalModel(LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(5, 1)
        
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def minimal_reproduce():
    # Create minimal synthetic data
    X = torch.randn(1000, 5)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Create model
    model = MinimalModel(lr=1e-5)
    
    # Create SWA callback - this is the problematic callback
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    
    # Create trainer with SWA callback
    trainer = Trainer(
        max_epochs=10,
        callbacks=[swa_callback],
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False
    )
    
    # Create tuner and run lr_find - this should trigger the error
    tuner = Tuner(trainer)
    
    lr_finder = tuner.lr_find(model, train_dataloaders=dataloader)
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {suggested_lr}")
    print("-"*100)
    model.hparams.lr = suggested_lr

    trainer.fit(model, train_dataloaders=dataloader)

if __name__ == "__main__":
    print("=== Minimal Reproduction of Issue #20070 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Lightning version: {L.__version__}")
    minimal_reproduce()
```
