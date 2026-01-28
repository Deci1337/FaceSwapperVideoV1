"""
Models for Hair Transfer Pipeline

Contains loaders for Stable Diffusion, ControlNet, IP-Adapter.
"""

from app.hair_transfer.models.sd_loader import SDInpaintLoader
from app.hair_transfer.models.pose_detector import PoseDetector

__all__ = ['SDInpaintLoader', 'PoseDetector']

