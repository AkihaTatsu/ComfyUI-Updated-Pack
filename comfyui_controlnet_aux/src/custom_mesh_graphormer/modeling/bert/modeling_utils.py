# Handle transformers version compatibility for utility functions
# These functions were in modeling_utils < 4.x, moved to pytorch_utils in 4.x, removed in 5.0.0+
import torch

try:
    # Try transformers < 5.0.0 (pytorch_utils exists)
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )
except ImportError:
    try:
        # Try older transformers where these are in modeling_utils
        from transformers.modeling_utils import (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )
    except ImportError:
        # transformers 5.0.0+ - provide fallback implementations
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
            """Apply chunking to a forward function if chunk_size > 0."""
            if chunk_size > 0:
                tensor_shape = input_tensors[0].shape[chunk_dim]
                if tensor_shape % chunk_size != 0:
                    raise ValueError(
                        f"The dimension {chunk_dim} of the input tensor with shape {tensor_shape} is not a multiple of chunk_size {chunk_size}"
                    )
                num_chunks = tensor_shape // chunk_size
                input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
                output_chunks = [forward_fn(*chunk) for chunk in zip(*input_chunks)]
                return torch.cat(output_chunks, dim=chunk_dim)
            return forward_fn(*input_tensors)
        
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            """Find heads and their indices to prune."""
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        
        def prune_linear_layer(layer, index, dim=0):
            """Prune a linear layer by keeping only the specified indices."""
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone().detach()
            if layer.bias is not None:
                if dim == 0:
                    b = layer.bias[index].clone().detach()
                else:
                    b = layer.bias.clone().detach()
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(W.contiguous())
            new_layer.weight.requires_grad = True
            if layer.bias is not None:
                new_layer.bias.requires_grad = False
                new_layer.bias.copy_(b.contiguous())
                new_layer.bias.requires_grad = True
            return new_layer

# Also import common utilities from transformers.modeling_utils that may still be needed
from transformers.modeling_utils import *

# Import Conv1D from pytorch_utils (transformers 4.x+) or provide fallback
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    # Conv1D fallback for older versions
    class Conv1D(torch.nn.Module):
        """
        1D convolution layer, similar to OpenAI GPT's implementation.
        Basically works like a Linear layer but the weights are transposed.
        """
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = torch.nn.Parameter(torch.empty(nx, nf))
            self.bias = torch.nn.Parameter(torch.zeros(nf))
            torch.nn.init.normal_(self.weight, std=0.02)

        def forward(self, x):
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            x = x.view(size_out)
            return x

# Import PretrainedConfig from transformers top-level
from transformers import PretrainedConfig

# Handle TF_WEIGHTS_NAME compatibility (removed in transformers 5.0.0)
try:
    from transformers.modeling_utils import TF_WEIGHTS_NAME
except ImportError:
    # TF_WEIGHTS_NAME was removed in transformers 5.0.0
    TF_WEIGHTS_NAME = "model.ckpt"

# prune_layer is an alias for prune_linear_layer in older code
prune_layer = prune_linear_layer
