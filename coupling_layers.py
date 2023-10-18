"""This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA.

Defines the coupling blocks for the normalizing flow.
"""


from typing import Callable, List, Optional, Tuple, Union

import torch
from FrEIA.modules.coupling_layers import _BaseCouplingBlock


class CouplingBlock(_BaseCouplingBlock):
    """Coupling Block following the GLOW design.

    Note, this is only the coupling
    part itself, and does not include ActNorm, invertible 1x1 convolutions, etc.
    See AllInOneBlock for a block combining these functions at once.
    The only difference to the RNVPCouplingBlock coupling blocks
    is that it uses a single subnetwork to jointly predict [s_i, t_i], instead of two separate
    subnetworks. This reduces computational cost and speeds up learning.
    """

    def __init__(
        self,
        dims_in: Union[int, Tuple[int], List[int]],
        dims_c: Optional[Union[int, Tuple[int], List[int]]] = None,
        subnet_constructor: Callable = lambda: None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
    ) -> None:
        """Additional args in docstring of base class.

        Args:
            dims_in: The input dimensions.
            dims_c: The coupling dimensions.
            subnet_constructor: function or class, with signature
                constructor(dims_in, dims_out).  The result should be a torch
                nn.Module, that takes dims_in input channels, and dims_out output
                channels. See tutorial for examples. Two of these subnetworks will be
                initialized in the block.
            clamp: Soft clamping for the multiplicative component. The
                amplification or attenuation of each input dimension can be at most
                exp(±clamp).
            clamp_activation: Function to perform the clamping. String values
                "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
                object can be passed. TANH behaves like the original realNVP paper.
                A custom function should take tensors and map -inf to -1 and +inf to +1.
        """
        dimensions_c = [] if dims_c is None else dims_c
        super().__init__(dims_in, dimensions_c, clamp, clamp_activation)

        self.subnet1 = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * 2)
        self.subnet2 = subnet_constructor(self.split_len2 + self.condition_length, self.split_len1 * 2)

    def _coupling1(self, x1: torch.Tensor, u2: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs first coupling.

        Args:
            x1: The inputs i.e. 'x-side' when rev is False, 'z-side' when rev is True.
            u2: _description_
            rev: Perform reverse computation if rev is True. Defaults to False.

        Returns:
            The results of the first coupling.
        """
        affine_2 = self.subnet2(u2)
        scale_2, translation_2 = affine_2[:, : self.split_len1], affine_2[:, self.split_len1 :]
        scale_2 = self.clamp * self.f_clamp(scale_2)
        jacobian_1 = scale_2

        if rev:
            outputs_1 = (x1 - translation_2) * torch.exp(-scale_2)
            return outputs_1, -jacobian_1

        outputs_1 = torch.exp(scale_2) * x1 + translation_2
        return outputs_1, jacobian_1

    def _coupling2(self, x2: torch.Tensor, u1: torch.Tensor, rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the second coupling.

        Args:
            x2: The inputs i.e. 'x-side' when rev is False, 'z-side' when rev is True.
            u1: _description_
            rev: Perform reverse computation if rev is True. Defaults to False.

        Returns:
            The results of the second coupling.
        """
        affine_1 = self.subnet1(u1)
        scale_1, transaltion_1 = affine_1[:, : self.split_len2], affine_1[:, self.split_len2 :]
        scale_1 = self.clamp * self.f_clamp(scale_1)
        jacobian_2 = scale_1

        if rev:
            outputs_2 = (x2 - transaltion_1) * torch.exp(-scale_1)
            return outputs_2, -jacobian_2

        outputs_2 = torch.exp(scale_1) * x2 + transaltion_1
        return outputs_2, jacobian_2
