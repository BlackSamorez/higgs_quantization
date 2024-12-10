import torch

from transformers.integrations.higgs import pad_to_block, get_higgs_grid


def get_grid_and_norm(bits: int, p: int, device: torch.device) -> torch.Tensor:
    grid = get_higgs_grid(p, 2**(p * bits))
    grid_norm_2 = torch.linalg.norm(grid, axis=-1) ** 2
    return grid.to(device), grid_norm_2.to(device)


def higgs_quantize(x: torch.Tensor, grid: torch.Tensor, grid_norm_2: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == grid.shape[-1], "Last dimension of x and grid must match"
    return torch.argmax(2 * x @ grid.T - grid_norm_2, dim=-1).to(torch.uint8)


def higgs_dequantize(q: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return grid[q.to(torch.int32)]
