---
title: Tuner
type: docs
prev: docs/01-basics/03-lightning/02-pytorch-lightning.md
sidebar:
  open: false
weight: 23
---

- `Tuner` is a utility in PyTorch Lightning that helps you `find the optimal batch size` and `learning rate` for your model training.

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
