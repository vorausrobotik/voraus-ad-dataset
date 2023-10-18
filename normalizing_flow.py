"""Contains the normalizing flow model and its internal network."""

from typing import Any, Dict, Tuple, Type

import numpy
import torch
from FrEIA.framework import InputNode, Node, OutputNode
from FrEIA.modules import PermuteRandom
from torch import Tensor, nn

from configuration import Configuration
from coupling_layers import CouplingBlock
from graph_inn import GraphINN


def get_loss(z_space: Tensor, jac: Tensor) -> Tensor:
    """Calculate the loss of a batch.

    Computes the negative log likelihood loss (per dimension) assuming z should be Gaussian.

    Args:
        z_space: The batch result.
        jac: The jacobian matrix.

    Returns:
        The loss of the batch.
    """
    sum_dimension = tuple(range(1, z_space.dim()))
    number = numpy.prod(z_space.shape[1:])
    return torch.mean(torch.sum(z_space**2, dim=sum_dimension) - jac) / number


def get_loss_per_sample(z_space: Tensor, jac: Tensor) -> Tensor:
    """Calculates the loss per sample.

    Args:
        z_space: The batch result.
        jac: The jacobian matrix.

    Returns:
        The loss per sample.
    """
    sum_dimension = tuple(range(1, z_space.dim()))
    loss = 0.5 * torch.sum(z_space**2, dim=sum_dimension) - jac
    return loss


class InternalNetwork(torch.nn.Module):
    """Describes the internal network used for the normalizing flow."""

    args: Tuple = tuple()
    kwargs: Dict[str, Any] = {}

    def __init__(  # pylint: disable=too-many-locals
        self,
        dims_in: int,
        dims_out: int,
        number_of_time_steps: int,
        number_of_signals: int,
        n_hidden_layers: int,
        scale: int,
        kernel_size_1: int,
        dilation_1: int,
        kernel_size_2: int,
        dilation_2: int,
        kernel_size_3: int,
        dilation_3: int,
    ):
        """Initializes a new internal network.

        Args:
            dims_in: The input dimensions.
            dims_out: The output dimensions.
            number_of_time_steps: The number of timestamps.
            number_of_signals: The number of signals.
            n_hidden_layers: The number of hidden layers.
            scale: The scale of the network.
            kernel_size_1: The kernel size of the first convolution.
            dilation_1: The dilation of the first convolution.
            kernel_size_2: The kernel size of the hidden convolutions.
            dilation_2: The dilation of the hidden convolutions.
            kernel_size_3: The kernal size of the last convolution.
            dilation_3: The dilation of the last convolution.
        """
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out

        self.T = number_of_time_steps  # pylint: disable=invalid-name
        self.dx = number_of_signals  # pylint: disable=invalid-name

        hidden_layers = torch.nn.ModuleList()
        for _ in range(n_hidden_layers):
            hidden_layers.extend(
                [
                    nn.Conv1d(
                        self.dx * scale,
                        self.dx * scale,
                        kernel_size=kernel_size_2,
                        dilation=dilation_2,
                        padding="same",
                        padding_mode="replicate",
                    ),
                    nn.ReLU(),
                ]
            )

        chn_in = self.dx // 2
        chn_out = self.dx

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                chn_in,
                self.dx * scale,
                kernel_size=kernel_size_1,
                dilation=dilation_1,
                padding="same",
                padding_mode="replicate",
            ),
            nn.ReLU(),
            *hidden_layers,
            nn.Conv1d(
                self.dx * scale,
                chn_out,
                kernel_size=kernel_size_3,
                dilation=dilation_3,
                padding="same",
                padding_mode="replicate",
            ),
        )

    @classmethod
    def setup(cls, *args: Any, **kwargs: Any) -> Type["InternalNetwork"]:
        """This method is used to create a new instance with the given parameters.

        Args:
            *args: The arguments for the TS network.
            **kwargs: The keyword arguments for the TS network.

        Returns:
            A new initialized TS network.
        """
        cls.args = args
        cls.kwargs = kwargs
        return cls

    @classmethod
    def constructor(cls, dims_in: int, dims_out: int) -> "InternalNetwork":
        """The abstract subnet constructor for the FrEYA coupling blocks.

        This method must be overriden by the inheriting class.

        Args:
            dims_in: The input dimensions.
            dims_out: The output dimensions.

        Returns:
            The initialized TS network.
        """
        return cls(dims_in, dims_out, *cls.args, **cls.kwargs)

    def forward(self, x: torch.Tensor) -> Any:  # pylint: disable=invalid-name
        """Forward computation of the internal network.

        Args:
            x: The batch.

        Returns:
            The latent space.
        """
        outputs = self.layer1(x)
        return outputs


class NormalizingFlow(GraphINN):
    """Describes the normalizing flow model."""

    def __init__(self, input_dimension: Tuple[int, ...], config: Configuration) -> None:
        """Initializes the normalizing flow model.

        Args:
            input_dimension: The input dimensions.
            config: The configuration of the model.
        """
        nodes = [InputNode(*input_dimension, name="input")]

        int_network = InternalNetwork.setup(
            input_dimension[1],
            input_dimension[0],
            n_hidden_layers=config.n_hidden_layers,
            scale=config.scale,
            kernel_size_1=config.kernel_size_1,
            dilation_1=config.dilation_1,
            kernel_size_2=config.kernel_size_2,
            dilation_2=config.dilation_2,
            kernel_size_3=config.kernel_size_3,
            dilation_3=config.dilation_3,
        )

        for cbi in range(config.n_coupling_blocks):
            kwargs: Dict[Any, Any] = {}

            nodes.append(
                Node(nodes[-1], PermuteRandom, kwargs, name=f"permute{cbi}"),
            )
            nodes.append(
                Node(
                    nodes[-1],
                    CouplingBlock,
                    {
                        "subnet_constructor": int_network.constructor,
                        "clamp": config.clamp,
                    },
                    name=f"cb{cbi}",
                )
            )

        output_node = OutputNode(nodes[-1], name="output")
        nodes.append(output_node)

        super().__init__(nodes)
