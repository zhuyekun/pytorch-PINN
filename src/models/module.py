from collections.abc import Callable, Sequence

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self,
        units: Sequence[int],
        activation: Callable = nn.ReLU,
        out_act: bool = True,
        name="mlp",
        **kwargs
    ):
        super(Mlp, self).__init__()

        num_layers = len(units) - 1
        assert num_layers >= 1, "num_layers must be greater than or equal to 1"

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(units[i], units[i + 1]))
            self.layers.append(activation())

        self.layers.append(nn.Linear(units[-2], units[-1]))
        if out_act:
            self.layers.append(activation())

        self.apply(_init_weights)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        units: Sequence[int],
        activation: Callable = nn.ReLU,
        out_act: bool = True,
        name="resnet`",
        **kwargs
    ):
        super(ResNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(units[0], units[1]))

        for i in range(1, len(units) - 1, 2):
            self.layers.append(
                ResBlock(
                    units=units[i : i + 2],
                    activation=activation(),
                )
            )

        self.layers.append(nn.Linear(units[-2], units[-1]))
        if out_act:
            self.layers.append(activation())

        self.apply(_init_weights)

    def forward(self, inputs):
        output = inputs

        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output


class ResBlock(nn.Module):
    def __init__(
        self,
        units: Sequence[int],
        activation: Callable = nn.ReLU,
        name="residual_block",
        **kwargs
    ):
        super(ResBlock, self).__init__()

        assert units[0] == units[-1], "First and last units must be the same"
        assert len(units) - 1 >= 1, "num_layers must be greater than or equal to 1"

        self._activation = activation

        self._layers = nn.ModuleList(
            [nn.Linear(units[i], units[i + 1]) for i in range(len(units) - 1)]
        )

    def forward(self, inputs):
        residual = inputs
        for h_i in self._layers:
            inputs = self._activation(h_i(inputs))
        residual = residual + inputs
        return residual


@torch.no_grad()
def _init_weights(module: nn.Module):
    """weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif hasattr(module, "init_weights"):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()
