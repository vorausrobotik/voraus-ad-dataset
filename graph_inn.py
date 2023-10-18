"""This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA.

In order to access the full jacobian, we have modified the GraphINN module.
"""


from typing import Any, Dict, List, Tuple, Union

import torch
from FrEIA.framework.graph_inn import GraphINN as OriginalGraphINN
from FrEIA.framework.graph_inn import Node
from torch import Tensor


class GraphINN(OriginalGraphINN):
    """This class represents the invertible net itself.

    It is a subclass of InvertibleModule and supports the same methods.
    The forward method has an additional option 'rev', with which the net can be
    computed in reverse. Passing `jac` to the forward method additionally
    computes the log determinant of the (inverse) Jacobian of the forward
    (backward) pass.
    """

    def forward(  # pylint: disable=too-complex,too-many-locals,too-many-branches,too-many-statements,arguments-differ
        self,
        x_or_z: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        rev: bool = False,
        jac: bool = True,
        intermediate_outputs: bool = False,
    ) -> Any:
        """Forward computation of the whole net.

        Args:
            x_or_z: The inputs of the net.
            rev: Reverse computation if True. Defaults to False.
            jac: Individual Jacobian computation if True. Defaults to True.
            intermediate_outputs: Return internal all Jacobians. Defaults to False.

        Raises:
            ValueError: If input or conditions shape does not match.
            RuntimeError: If module computation fails.

        Returns:
            Any: The outputs and the Jacobian.
        """
        if isinstance(x_or_z, torch.Tensor):
            x_or_z = (x_or_z,)

        x_or_z_length = len(x_or_z)

        jacobian = 0
        outs: Dict[Any, Tensor] = {}
        jacobian_dict: Dict[Node, torch.Tensor] = {}

        # Explicitly set conditions and starts
        start_nodes = self.out_nodes if rev else self.in_nodes
        start_nodes_length = len(start_nodes)
        if x_or_z_length != start_nodes_length:
            raise ValueError(f"Got {x_or_z_length} inputs, but expected " f"{start_nodes_length}.")
        for tensor, start_node in zip(x_or_z, start_nodes):
            outs[start_node, 0] = tensor

        # Go backwards through nodes if rev=True
        for node in self.node_list[:: -1 if rev else 1]:
            # Skip all special nodes
            if node in self.in_nodes + self.out_nodes + self.condition_nodes:
                continue

            mod_in: List = []
            for prev_node, channel in node.outputs if rev else node.inputs:
                mod_in.append(outs[prev_node, channel])
            mod_in_tuple = tuple(mod_in)

            try:
                mod_out = node.module(mod_in_tuple, rev=rev, jac=jac)
            except Exception as error:
                raise RuntimeError(f"{node} encountered an error.") from error

            out, mod_jac = self._check_output(node, mod_out, jac, rev)

            for out_idx, out_value in enumerate(out):
                outs[node, out_idx] = out_value

            if jac and not node.name.startswith("permute"):
                jacobian = jacobian + mod_jac
                jacobian_dict[node] = mod_jac

        for out_node in self.in_nodes if rev else self.out_nodes:
            # This copies the one input of the out node
            outs[out_node, 0] = outs[(out_node.outputs if rev else out_node.inputs)[0]]

        if intermediate_outputs:
            return outs, jacobian_dict

        out_list = [outs[out_node, 0] for out_node in (self.in_nodes if rev else self.out_nodes)]
        if len(out_list) == 1 and not self.force_tuple_output:
            return out_list[0], jacobian

        return tuple(out_list), jacobian
