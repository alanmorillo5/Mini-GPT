# Checkpoints Directory

This directory stores the saved state dictionaries (model weights) generated during the training process. 

## Checkpoint Schema

The models are saved utilizing PyTorch's `torch.save` mechanism to a dictionary format containing both the model state and the optimizer state. By default, the best model is saved as `ckpt.pt`.

When a checkpoint is saved, its schema looks like this:

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter_num': iter_num,
    'best_val_loss': best_val_loss
}
```

### Key Components:
- **`model`**: Contains the `state_dict()` of the GPT network, which maps each layer (like the RoPE embeddings, Multi-Query Attention weights, and SwiGLU MLP layers) to its tensor weights.
- **`optimizer`**: The `state_dict()` of the AdamW optimizer. This is saved so you can resume training smoothly without losing the momentum and variance tracking states.
- **`iter_num`**: The specific iteration/step number at which the checkpoint was saved.
- **`best_val_loss`**: The lowest validation loss achieved up to that point. The trainer uses this metric to ensure it only saves the checkpoint if the model improves.

## Loading a Checkpoint

To manually load a checkpoint in Python for custom inference or resuming training:

```python
import torch

device = 'mps' # Or 'cpu' / 'cuda'
checkpoint = torch.load('checkpoints/ckpt.pt', map_location=device)

# Load model weights
model.load_state_dict(checkpoint['model'])

# If resuming training, also load the optimizer state
optimizer.load_state_dict(checkpoint['optimizer'])
```

*Note: The CLI wrapper `main.py` automatically handles loading the `ckpt.pt` checkpoint for you when running in `--chat` mode.*
