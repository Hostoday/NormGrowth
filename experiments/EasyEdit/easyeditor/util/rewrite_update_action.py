"""Apply a stored rewrite-parameter update to column-oriented MLP keys."""
import torch
from transformers.pytorch_utils import Conv1D


def rewrite_update_action(module, stored_update, keys):
    """Return the output change, respecting Linear versus GPT2 Conv1D storage."""
    if isinstance(module, Conv1D):
        output_by_input = stored_update.T
    elif isinstance(module, torch.nn.Linear):
        output_by_input = stored_update
    else:
        raise TypeError(f"Unsupported rewrite module: {type(module).__name__}")
    return output_by_input.to(device=keys.device, dtype=keys.dtype) @ keys
