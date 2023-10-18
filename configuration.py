"""Contains the configuration model description."""


from typing import Literal

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Describes the configuration parameters."""

    seed: int
    epochs: int
    batchsize: int
    n_hidden_layers: int = Field(alias="nHiddenLayers")
    n_coupling_blocks: int = Field(alias="nCouplingBlocks")
    scale: int
    columns: Literal["machine", "mechanical", "electrical", "computed", "measured"]
    clamp: float
    pad: bool
    frequency_divider: int = Field(alias="frequencyDivider")
    train_gain: float = Field(alias="trainGain")
    normalize: bool
    kernel_size_1: int = Field(alias="kernelSize1")
    dilation_1: int = Field(alias="dilation1")
    kernel_size_2: int = Field(alias="kernelSize2")
    dilation_2: int = Field(alias="dilation2")
    kernel_size_3: int = Field(alias="kernelSize3")
    dilation_3: int = Field(alias="dilation3")
    milestones: list[int]
    gamma: float
    learning_rate: float = Field(alias="learningRate")
