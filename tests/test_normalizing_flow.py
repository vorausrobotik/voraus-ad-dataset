"""Contains tests for the normalizing flow module."""


from typing import List

import pytest
import torch

from configuration import Configuration
from normalizing_flow import InternalNetwork, NormalizingFlow, get_loss, get_loss_per_sample


@pytest.mark.parametrize(
    ("z_tensor", "jacobian", "expected_loss"),
    (
        ([[0, 1, 2, 3], [2, 3, 4, 5]], [[1, 3], [1, 3]], 8.0),
        ([[1, 2, 3, 0], [4, 3, 2, 5]], [[1, 3], [1, 3]], 8.0),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[1, 3], [1, 3]], 10.8750),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[1, 3], [3, 1]], 10.8750),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[0, 2], [3, 5]], 10.75),
    ),
)
def test_get_loss(z_tensor: List[float], jacobian: List[float], expected_loss: float) -> None:
    loss = get_loss(torch.Tensor(z_tensor), torch.Tensor(jacobian))
    assert loss.item() == pytest.approx(expected_loss, abs=1e-3)


@pytest.mark.parametrize(
    ("z_tensor", "jacobian", "expected_loss"),
    (
        ([[0, 1, 2, 3], [2, 3, 4, 5]], [[3, 4], [1, 3]], [[4, 23], [6, 24]]),
        ([[1, 2, 3, 0], [4, 3, 2, 5]], [[1, 3], [1, 3]], [[6, 24], [6, 24]]),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[1, 3], [1, 3]], [[19.5, 22], [19.5, 22]]),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[1, 3], [3, 1]], [[19.5, 22], [17.5, 24]]),
        ([[6, 0, 1, 2], [7, 0, 0, 1]], [[0, 2], [3, 5]], [[20.5, 23], [17.5, 20]]),
    ),
)
def test_get_loss_per_sample(z_tensor: List[float], jacobian: List[float], expected_loss: List[float]) -> None:
    loss = get_loss_per_sample(torch.Tensor(z_tensor), torch.Tensor(jacobian))
    for sample_i in range(len(z_tensor)):
        assert loss[sample_i].detach() == pytest.approx(expected_loss[sample_i], abs=1e-3)


def test_internal_network() -> None:
    internal_network_factory = InternalNetwork.setup(
        number_of_time_steps=3,
        number_of_signals=2,
        n_hidden_layers=1,
        scale=1,
        kernel_size_1=3,
        kernel_size_2=1,
        kernel_size_3=3,
        dilation_1=1,
        dilation_2=1,
        dilation_3=5,
    )

    internal_network = internal_network_factory.constructor(3, 3)
    modules = list(internal_network.modules())

    assert isinstance(modules[0], InternalNetwork)
    assert isinstance(modules[1], torch.nn.Sequential)

    assert isinstance(modules[2], torch.nn.Conv1d)
    assert modules[2].in_channels == 1
    assert modules[2].out_channels == 2
    assert modules[2].kernel_size == (3,)
    assert modules[2].dilation == (1,)

    assert isinstance(modules[3], torch.nn.ReLU)

    assert isinstance(modules[4], torch.nn.Conv1d)
    assert modules[4].in_channels == 2
    assert modules[4].out_channels == 2
    assert modules[4].kernel_size == (1,)
    assert modules[4].dilation == (1,)

    assert isinstance(modules[5], torch.nn.ReLU)

    assert isinstance(modules[6], torch.nn.Conv1d)
    assert modules[6].in_channels == 2
    assert modules[6].out_channels == 2
    assert modules[6].kernel_size == (3,)
    assert modules[6].dilation == (5,)

    x_tensor = torch.Tensor([[0, 2, 3]])
    y_tensor = internal_network.forward(x_tensor)

    assert y_tensor.shape == (2, 3)


def test_normalizing_flow() -> None:
    configuration = Configuration(
        columns="machine",
        epochs=70,
        frequencyDivider=1,
        trainGain=1.0,
        seed=177,
        batchsize=32,
        nCouplingBlocks=2,
        clamp=1.2,
        learningRate=8e-4,
        normalize=True,
        pad=True,
        nHiddenLayers=0,
        scale=2,
        kernelSize1=3,
        dilation1=2,
        kernelSize2=1,
        dilation2=1,
        kernelSize3=1,
        dilation3=1,
        milestones=[11, 61],
        gamma=0.1,
    )
    normalizing_flow = NormalizingFlow((2, 3), configuration)

    assert [m.__class__.__name__ for m in normalizing_flow.modules()] == [
        "NormalizingFlow",
        "ModuleList",
        "PermuteRandom",
        "CouplingBlock",
        "InternalNetwork",
        "Sequential",
        "Conv1d",
        "ReLU",
        "Conv1d",
        "InternalNetwork",
        "Sequential",
        "Conv1d",
        "ReLU",
        "Conv1d",
        "PermuteRandom",
        "CouplingBlock",
        "InternalNetwork",
        "Sequential",
        "Conv1d",
        "ReLU",
        "Conv1d",
        "InternalNetwork",
        "Sequential",
        "Conv1d",
        "ReLU",
        "Conv1d",
    ]

    x_tensor = torch.Tensor([[[0, 2, 3], [1, 4, 5]]])
    z_tensor, jacobian = normalizing_flow.forward(x_tensor)

    assert z_tensor[0].shape == (2, 3)
    assert jacobian[0].shape == (1, 3)
