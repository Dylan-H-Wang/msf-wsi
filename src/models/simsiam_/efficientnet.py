import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from torchvision.models import efficientnet


class EfficientNet(efficientnet.EfficientNet):
    def __init__(
        self,
        inverted_residual_setting,
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            inverted_residual_setting,
            dropout,
            stochastic_depth_prob,
            num_classes,
            norm_layer,
            last_channel,
            **kwargs
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        feats = [None] * len(self.features)

        for i, layer in enumerate(self.features):
            x = layer(x)
            feats[i] = torch.flatten(self.avgpool(x), 1)
        feats.pop(0)

        x = self.classifier(feats[-1])

        return feats
