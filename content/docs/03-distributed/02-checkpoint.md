---
title: Distributed Checkpoint
weight: 302
---

DCP support loading & saving models from multiple ranks in parallel. It handles load-time resharding which enables saving in one cluster topology and loading into another.

## Code Example to load and save

```python
# filename: fsdp_checkpoint_example.py

# command to run (`t4 with 4gpu machine`): torchrun \
#       --nnodes=1 \
#       --nproc-per-node=4 \
#       --rdzv-id=1721 \
#       fsdp_checkpoint_example.py

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

CHECKPOINT_DIR = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = fully_shard(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    # Fixed input for reproducibility
    torch.manual_seed(42)  # Ensure same input across runs
    test_input = torch.arange(32, dtype=torch.float32).reshape(2, 16).to("cuda")
    
    with torch.no_grad():
        output_before = model(test_input)
        print(f"[Rank {rank}] Output BEFORE save: {output_before}")

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("Checkpoint saved. Now loading to verify...")
        print(f"{'='*60}\n")

    return test_input, output_before


def run_fsdp_checkpoint_load_example(rank, world_size, test_input, output_before):
    print(f"Loading checkpoint on rank {rank}.")
    
    # Create a fresh model (with random weights)
    model = ToyModel().to(rank)
    model = fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Verify fresh model gives different output
    with torch.no_grad():
        output_fresh = model(test_input)
        print(f"[Rank {rank}] Output from FRESH model: {output_fresh}")
    
    # Load checkpoint
    state_dict = { "app": AppState(model, optimizer) }
    dcp.load(state_dict, checkpoint_id=CHECKPOINT_DIR)
    
    # Test with same input
    with torch.no_grad():
        output_after = model(test_input)
        print(f"[Rank {rank}] Output AFTER load:  {output_after}")
    
    # Compare
    if torch.allclose(output_before, output_after, rtol=1e-5, atol=1e-7):
        print(f"[Rank {rank}] ✅ Outputs MATCH! Checkpoint loaded correctly.")
    else:
        max_diff = (output_before - output_after).abs().max().item()
        print(f"[Rank {rank}] ❌ Outputs DIFFER! Max difference: {max_diff:.2e}")
        print(f"[Rank {rank}]    Before: {output_before}")
        print(f"[Rank {rank}]    After:  {output_after}")
    
    return output_after


class Pipeline:
    def __init__(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)
    
    def run(self):
        self.setup()
        self.execute()
        self.teardown()
    
    def setup(self):
        self._device_mesh = dist.device_mesh.init_device_mesh("cuda", (self.world_size,))
        print(f"{self._device_mesh=}")

    def execute(self):
        # Save checkpoint and get reference output
        test_input, output_before = run_fsdp_checkpoint_save_example(
            self.global_rank, self.world_size
        )
        
        # Barrier to ensure all ranks finish saving
        dist.barrier()
        
        # Load checkpoint and verify
        output_after = run_fsdp_checkpoint_load_example(
            self.global_rank, self.world_size, test_input, output_before
        )

    def teardown(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
```
