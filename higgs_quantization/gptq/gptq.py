from math import sqrt
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

import flute.utils
from flute.integrations.higgs import prepare_data_transposed
from fast_hadamard_transform import hadamard_transform

from ..higgs.quantization import higgs_quantize, higgs_dequantize, get_grid_and_norm, pad_to_block


@torch.no_grad()
def gptq_block(
    block_weight: Tensor, block_hessian_inverse: Tensor,
    grid: Tensor, grid_norm_2: Tensor
) -> tuple[Tensor, Tensor]:
    p = grid.shape[1]
    
    quantized_block_weight = torch.zeros(
        (block_weight.shape[0], block_weight.shape[1]//p), device=block_weight.device, dtype=torch.uint8,
    )
    scaled_block_error = torch.zeros_like(block_weight)

    # Interate over the block's columns
    assert block_weight.shape[1] % p == 0
    
    for i in range(0, block_weight.shape[1], p):
        # Get the column and the corresponding inverse Hessian
        column_weight = block_weight[:, i:i+p]
        column_hessian_inverse = torch.diag(block_hessian_inverse)[i:i+p]

        # Quantize the column weight
        quantized_column_weight = higgs_quantize(column_weight, grid, grid_norm_2)
            
        quantized_block_weight[:, i//p] = quantized_column_weight
        dequantized_column_weight = higgs_dequantize(quantized_column_weight, grid)

        # Update all the following columns within the block
        scaled_column_error = (column_weight - dequantized_column_weight) / column_hessian_inverse
        block_weight[:, i+1:] -= scaled_column_error.matmul(block_hessian_inverse[i:i+p, i+1:])
        scaled_block_error[:, i:i+p] = scaled_column_error

    return quantized_block_weight, scaled_block_error, block_weight


def prepare_inverse_hessian(hessian: Tensor, percdamp: float) -> Tensor:
    """Precomputes inverse Hessian
    Args:
        hessian (Tensor): problem hessian
        percdamp (float): diagonal damping constant for numerical stability
    Returns:
        Tensor: precomputed inverse Hessian
    """
    damp = percdamp * torch.mean(torch.diag(hessian))
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    hessian[diag, diag] += damp
    hessian = torch.linalg.cholesky(hessian)
    hessian = torch.cholesky_inverse(hessian)
    hessian = torch.linalg.cholesky(hessian, upper=True)
    return hessian


@torch.no_grad()
def apply_higgs_gptq(
    weight: torch.Tensor, hessian: torch.Tensor,
    p: int, bits: int, hadamard_size: int = 512, group_size: int = 256,
    percdamp: float = .01,
) -> tuple[Tensor, Tensor, Tensor]:
    blocksize = p
    while blocksize < 128:
        blocksize *= 2

    dtype = weight.dtype
    weight = weight.float()
    num_columns = weight.shape[1]
    hessian = hessian.float()

    weight = pad_to_block(weight, [1], hadamard_size)
    hessian = pad_to_block(hessian, [0, 1], hadamard_size)
    
    # Scale and Hadamard transform weight
    mult = weight.shape[1] // hadamard_size
    weight = weight.reshape(-1, mult, hadamard_size)
    scales = torch.linalg.norm(weight, axis=-1, keepdim=True)
    weight = hadamard_transform(weight, 1) / scales
    weight = weight.reshape(weight.shape[0], -1)

    # Hadamard transform Hessian
    hessian = hessian.reshape(mult, hadamard_size, mult, hadamard_size)
    hessian = hadamard_transform(
        hadamard_transform(hessian, scale=1 / sqrt(hadamard_size)).permute(2, 3, 0, 1),
        scale=1 / sqrt(hadamard_size)
    ).permute(2, 3, 0, 1).reshape(weight.shape[1], weight.shape[1])

    # Process the Hessian to obtain the precomputed inverse Hessian
    hessian_inverse = prepare_inverse_hessian(hessian, percdamp)
    
    # Get the grid and its norm
    grid, grid_norm_2 = get_grid_and_norm(bits, p, weight.device)
    
    # Iterate over the columns in blocks
    assert weight.shape[1] % p == 0
    quantized_shape = (weight.shape[0], weight.shape[1] // p)
        
    codes = torch.empty(
        quantized_shape, dtype=torch.uint8, device=weight.device
    )

    for block_start in trange(0, num_columns, blocksize, leave=False, desc="GPTQ blocks..."):
        # YOUR CODE HERE>>>>>>>>>
        block_end = min(block_start + blocksize, weight.shape[1])

        # Get the next block and quantize it
        block_codes, block_error, weight[:, block_start:block_end] = gptq_block(
            weight[:, block_start:block_end], hessian_inverse[block_start:block_end, block_start:block_end],
            grid=grid, grid_norm_2=grid_norm_2
        )

        # Tune all the following blocks to mitigate the quantization error
        codes[:, block_start//p:block_end//p] = block_codes
        weight[:, block_end:] -= block_error.matmul(hessian_inverse[block_start:block_end, block_end:])
        # <<<<<<<<<<<<<<<<<<<<<<<
        
    codes = codes.reshape(codes.shape[0], -1)
    scales = scales.squeeze(-1) / sqrt(hadamard_size)
        
    weight, scales, tables, tables2 = prepare_data_transposed(
        codes,
        torch.repeat_interleave(scales.to(dtype), hadamard_size // group_size, dim=1),
        grid.to(dtype),
        num_bits=bits,
        group_size=group_size,
        vector_size=p,
        dtype=dtype,
        device=codes.device,
    )

    return {
        "weight": weight,
        "scales": scales,
        "tables": tables,
        "tables2": tables2.view(dtype=torch.float16),
    }


def get_accumulate_input_fn(name: str, hessians: Mapping[str, Tensor], num_samples: Mapping[str, int]):
    """Generate a callback that updates the corresponding hessians and counts when given input
    Args:
        name (str): module name
        hessians (Mapping[str, Tensor]): a dict of modules' hessians, accessible by module name
        num_samples (Mapping[str, int]): a dict of callback call counters
    """
    def tmp(_, inp, out):
        inp = inp[0].data # ... x hidden_size
        inp = inp.reshape((-1, inp.shape[-1])) # inputs x hidden_size
        inp = inp.t().float() # hidden_size x inputs
        num_samples[name] += inp.shape[1]
        if hessians[name] is None:
            hessians[name] = inp.matmul(inp.t())
        else:
            hessians[name] += inp.matmul(inp.t())
    return tmp