from collections.abc import Callable, Sequence

import torch.nn as nn

class Mlp(nn.Module):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        act_layer: Callable = nn.ReLU,
        out_act: bool = True,
    ):
        super().__init__()

        num_layers = len(layer_sizes) - 1
        assert num_layers >= 1, "num_layers must be greater than or equal to 1"

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            )
            self.layers.append(act_layer())

        self.layers.append(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        if out_act:
            self.layers.append(act_layer())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x