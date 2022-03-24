from typing import Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.utils import make_grid


def plot_imgs(imgs: Tensor, ncols: int = 1, fig_size : tuple = (10, 10)) -> None:
    imgs = imgs.detach().cpu()
    grid = make_grid(imgs, nrow=ncols)
    plt.figure(figsize=fig_size)
    plt.imshow(grid.permute(1, 2, 0), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

def minmax_normalize(
    x: Tensor, lower_bound: Optional[int] = None, upper_bound: Optional[int] = None
) -> Tensor:
    """
    Normalize a provided array to 0-1 range.
    Normalize a provided array to 0-1 range.
    :param x: Input array
    :param lower_bound: Optional lower bound. Takes minimum of x if not provided.
    :param upper_bound: Optional upper bound. Takes maximum of x if not provided.
    :return: Normalized array
    """
    if lower_bound is None:
        lower_bound = torch.min(x)

    if upper_bound is None:
        upper_bound = torch.max(x)

    return (x - lower_bound) / (upper_bound - lower_bound)
