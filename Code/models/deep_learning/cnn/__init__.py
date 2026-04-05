"""
Convolutional Neural Networks (CNN) package.
Reference: docs/deep-learning/cnn/cnn.md
"""

from ._base import CNNModuleBase
from .layers import (
    CapsNetsLayer,
    GroupEquivariantConvolutionalLayer,
    ShiftNetLayer,
    DeformableConvLayer,
    InvolutionLayer,
)
from .models import (
    CNNetworks,
    CNNetworksOp,
    CapsNetsModule,
    GroupEquivariantConvolutionalModule,
    ShiftNetsModule,
    DeformableConvModule,
    InvolutionModule,
    VolterraConvModule,
    DynamicSnakeConvModule,
    ODConvModule,
    ShiftwiseConvModule,
    SEAFECModule,
    IncoherentMotifModule,
)
from .dynamic import DynamicSnakeConvLayer, ODConvLayer
from .nonlinear import VolterraConvLayer
from .shiftwise import ShiftwiseConvLayer
from .capsule_kan import CapsuleConvKANLayer
from .spatial_edge import SEAFECLayer
from .structural import IncoherentMotifLayer
from .quantum import QCNNLayer
from .event import EventConvLayer
from .topological import TopologyNetLayer

__all__ = [
    "CNNModuleBase",
    "CNNetworks",
    "CNNetworksOp",
    "CapsNetsModule",
    "GroupEquivariantConvolutionalModule",
    "ShiftNetsModule",
    "DeformableConvModule",
    "InvolutionModule",
    "VolterraConvModule",
    "DynamicSnakeConvModule",
    "ODConvModule",
    "ShiftwiseConvModule",
    "SEAFECModule",
    "IncoherentMotifModule",
    "CapsNetsLayer",
    "GroupEquivariantConvolutionalLayer",
    "ShiftNetLayer",
    "DeformableConvLayer",
    "InvolutionLayer",
    "DynamicSnakeConvLayer",
    "ODConvLayer",
    "VolterraConvLayer",
    "ShiftwiseConvLayer",
    "CapsuleConvKANLayer",
    "SEAFECLayer",
    "IncoherentMotifLayer",
    "QCNNLayer",
    "EventConvLayer",
    "TopologyNetLayer",
]
