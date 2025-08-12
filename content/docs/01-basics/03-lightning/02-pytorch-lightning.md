---
title: PyTorch Lightning code
type: docs
prev: docs/01-basics/03-lightning/01-intro.md
sidebar:
  open: false
weight: 23
---

## Simple PL example

```python
import torch
import lightning as L

# -- dataset --
x = torch.randn(100,4)         # random dataset of size 100, containing 4 features
y = torch.randint(0,2, (100,)) # random labels: {0, 1} (2 classes)

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, drop_last=True)


# -- lightning module --

class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 8)
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Linear(8,2) # number of classes 2

    def forward(self, x):
        return self.out(self.act(self.layer(x))) 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),lr = 1e-3)
        return opt

# -- instantiate model --

model = SimpleModel()

# -- lightning trainer --

trainer = L.Trainer(max_epochs=10, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=dataloader)
```

---

## Using `test` & `validation` set

- define `test_step` & `validation_step` in **`LightningModule`**

> they don't return `loss` or anything

```python
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        self.log("loss", loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        self.log("loss", loss)
```

- **for test**, pass a `dataloader` and then call:
```python
trainer.test(model, test_dataloader)
```

- **for validation**, pass a `train dataloader` & a `val dataloader` and then call:
```python
trainer.fit(model, train_dataloader, val_dataloader)
```

---

## Saving Hyperparameters

```python
class LitAutoencoder(L.LightningModule):
    def __init__(self, encoder, decoder, num_latent_dims):
        self.encoder = encoder
        self.decoder = decoder
        self.num_latent_dims = num_latent_dims

        self.save_hyperparameters(ignore=["encoder", "decoder"])

model = LitAutoencoder.load_from_checkpoint(PATH, encoder=encoder, decoder=decoder)
```

- Now, when loading model from a checkpoint, it will automatically restore the hyperparameters.

- If some of the hyperparameters are too large to be saved, you can exclude them by passing `ignore` argument to `save_hyperparameters()`.

- ignore parameters via `save_hyperparameters(ignore=...)`, then you must pass the missing positional arguments or keyword arguments when calling `load_from_checkpoint` method

---

## Checkpointing

- Load state of a model for inferencing or predicting

```python
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)
```

- if you want to resume training, you can do so by calling:

```python
model = LitModel()
trainer = Trainer()

# automatically restores model, epoch, step, LR schedulers, etc...
trainer.fit(model, ckpt_path="path/to/your/checkpoint.ckpt")
```

---

## Callbacks

### EarlyStopping

- Stop training when a monitored metric has stopped improving.

```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)


model = LitModel()
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model)
```
