"""Contains tests for the graph inn module."""

import re
from unittest.mock import MagicMock

import pytest
import torch
from FrEIA.framework import InputNode, Node, OutputNode
from torch.nn import Linear
from torch.nn.parameter import Parameter

from coupling_layers import CouplingBlock
from graph_inn import GraphINN


def test_graph_inn_forward() -> None:
    x_tensor = torch.Tensor([[[0, 1, 0, 2]]])
    input_dimension = (x_tensor.shape[2],)
    split_len_1 = input_dimension[0] // 2
    split_len_2 = input_dimension[0] // 2

    subnet_1 = Linear(in_features=split_len_2, out_features=split_len_2 * 2, bias=False)
    subnet_2 = Linear(in_features=split_len_1, out_features=split_len_1 * 2, bias=False)

    with torch.no_grad():
        subnet_1.weight = Parameter(torch.ones_like(subnet_1.weight))
        subnet_2.weight = Parameter(torch.ones_like(subnet_2.weight))

    subnet_mock = MagicMock()
    subnet_mock.side_effect = [subnet_1, subnet_2]

    input_node = InputNode(*input_dimension, name="input")
    coupling_node = Node(
        input_node,
        CouplingBlock,
        {
            "subnet_constructor": subnet_mock,
            "clamp": 1,
        },
        name="cb",
    )
    output_node = OutputNode(coupling_node, name="output")

    graph_inn = GraphINN([input_node, coupling_node, output_node])

    # Test usual forward call.
    z_tensor, jacobian = graph_inn.forward(x_tensor[0])
    assert z_tensor[0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=1e-3)
    assert jacobian[0].detach() == pytest.approx([1.5985, 1.5985], abs=1e-3)

    # Test forward call with tuple.
    z_tensor, jacobian = graph_inn.forward((x_tensor[0],))
    assert z_tensor[0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=1e-3)
    assert jacobian[0].detach() == pytest.approx([1.5985, 1.5985], abs=1e-3)

    # Test forward call with intermediate outputs.
    z_dict, jacobian_dict = graph_inn.forward(x_tensor[0], intermediate_outputs=True)
    assert (input_node, 0) in z_dict
    assert (coupling_node, 0) in z_dict
    assert (output_node, 0) in z_dict
    assert z_dict[(input_node, 0)][0].detach() == pytest.approx([0, 1, 0, 2], abs=1e-3)
    assert z_dict[(coupling_node, 0)][0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=1e-3)
    assert z_dict[(output_node, 0)][0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=1e-3)
    assert coupling_node in jacobian_dict
    assert jacobian_dict[coupling_node][0].detach() == pytest.approx([1.5985, 1.5985], abs=1e-3)

    # Test forward call with force tuple output.
    graph_inn.force_tuple_output = True
    try:
        (z_tensor,), jacobian = graph_inn.forward(x_tensor[0])
    finally:
        graph_inn.force_tuple_output = False

    assert z_tensor[0].detach() == pytest.approx([2, 4.0221, 6.0221, 10.9137], abs=1e-3)
    assert jacobian[0].detach() == pytest.approx([1.5985, 1.5985], abs=1e-3)

    # Test forward call with too large x tensor.
    with pytest.raises(ValueError, match="Got 2 inputs, but expected 1."):
        graph_inn.forward(
            (
                x_tensor[0],
                x_tensor[0],
            )
        )

    # Test forward call with bad dimensions.
    with pytest.raises(
        RuntimeError, match=re.escape("Node 'cb': [(4,)] -> CouplingBlock -> [(4,)] encountered an error.")
    ):
        graph_inn.forward((x_tensor[:3]))
