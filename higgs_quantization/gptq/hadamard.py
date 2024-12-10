import math

import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform


def pad_to_block(tensor, dims, blocksize):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_block = ((size - 1) // blocksize + 1) * blocksize
        delta = next_multiple_of_block - size
        pad_dims[-2 * dim - 1] = delta
    
    return F.pad(tensor, pad_dims, "constant", 0)


class HadLinear(nn.Module):
    def __init__(self, weight, blocksize, do_hadamard):
        super().__init__()
        self.blocksize = blocksize
        self.do_hadamard = do_hadamard

        if do_hadamard:
            weight = weight / math.sqrt(blocksize)
        self.weight = nn.Parameter(weight)
    
    def forward(self, input):
        if self.do_hadamard:
            input = pad_to_block(input, [-1], self.blocksize)
            mult = input.shape[-1] // self.blocksize
            input = hadamard_transform(
                input.reshape(input.shape[:-1] + (mult, self.blocksize)),
                scale=1/math.sqrt(self.blocksize)
            )

            input = input.reshape(input.shape[:-2] + (mult * self.blocksize,))
            
        return F.linear(input, self.weight)
