"""S-Mamba 主干：复用 S-D-Mamba 开源实现，统一对外接口。"""
from .interface import BaseBackbone, BackboneOutput, MockBackbone
from .smamba_backbone import SMambaBackbone

__all__ = ["BaseBackbone", "BackboneOutput", "MockBackbone", "SMambaBackbone"]
