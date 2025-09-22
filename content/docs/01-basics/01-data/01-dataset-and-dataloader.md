---
title: Dataset & Dataloader
type: docs
prev: docs/01-basics/01-data
next: docs/01-basics/02-concepts
sidebar:
  open: false
weight: 12
---

Dataset & DataLoader

## Pin Memory

If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed up the host to device transfer by enabling pin_memory.

> This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.

```python
    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # validation dataloader
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
```
