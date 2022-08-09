import os
import torch
import torchvision

from torch import Tensor
from torch import nn
import kornia.augmentation as K



class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        
        self.transforms = K.VideoSequential(
            K.RandomAffine(360),
            K.ColorJiggle(0.2, 0.3, 0.2, 0.3),
            K.RandomMixUp(p=1.0),
            data_format="BCTHW",
            same_on_frame=True
        )

    @torch.no_grad() 
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxTxCxHxW
        return x_out