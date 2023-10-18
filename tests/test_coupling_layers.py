"""Contains tests for the coupling layers module."""


from unittest.mock import MagicMock, call

import pytest
import torch
from FrEIA.framework import InputNode, Node
from torch.nn import Linear
from torch.nn.parameter import Parameter

from coupling_layers import CouplingBlock


def test_coupling_block_subnet_consteruction() -> None:
    input_dimension = (3,)
    split_len_1 = 2
    split_len_2 = 1

    input_node = InputNode(*input_dimension, name="input")
    subnet_mock = MagicMock()

    node = Node(
        input_node,
        CouplingBlock,
        {
            "subnet_constructor": subnet_mock,
            "clamp": 1,
        },
        name="cb",
    )

    subnet_mock.assert_has_calls(
        [
            call(split_len_1, split_len_2 * 2),
            call(split_len_2, split_len_1 * 2),
        ]
    )

    assert isinstance(node.module, CouplingBlock)
    assert isinstance(node.module.subnet1, MagicMock)
    assert isinstance(node.module.subnet2, MagicMock)


def test_coupling_block_forward() -> None:
    x_tensor = torch.Tensor([[[0, 1, 0, 2]]])
    input_dimension = (x_tensor.shape[2],)
    split_len_1 = input_dimension[0] // 2
    split_len_2 = input_dimension[0] // 2

    input_node = InputNode(*input_dimension, name="input")

    subnet_1 = Linear(in_features=split_len_2, out_features=split_len_2 * 2, bias=False)
    subnet_2 = Linear(in_features=split_len_1, out_features=split_len_1 * 2, bias=False)

    with torch.no_grad():
        subnet_1.weight = Parameter(torch.ones_like(subnet_1.weight))
        subnet_2.weight = Parameter(torch.ones_like(subnet_2.weight))

    subnet_mock = MagicMock()
    subnet_mock.side_effect = [subnet_1, subnet_2]

    node = Node(
        input_node,
        CouplingBlock,
        {
            "subnet_constructor": subnet_mock,
            "clamp": 1,
        },
        name="cb",
    )

    assert isinstance(node.module, CouplingBlock)
    assert node.module.subnet1 is subnet_1
    assert node.module.subnet2 is subnet_2

    (y_tensor,), jacobian = node.module.forward(x_tensor)

    assert y_tensor[0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=0.001)
    assert jacobian[0].detach() == pytest.approx([1.5985, 1.5985], abs=0.001)

    (x_tensor_reconstruction,), jacobian = node.module.forward(y_tensor[None], rev=True)

    assert x_tensor_reconstruction[0].detach() == pytest.approx(x_tensor[0, 0, :], abs=0.001)
