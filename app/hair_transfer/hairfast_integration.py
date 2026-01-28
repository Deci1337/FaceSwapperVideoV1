"""
Thin wrapper for HairFastGAN integration.
Requires external/HairFastGAN (or installed package).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class HairFastWrapper:
    def __init__(self, device: str = "cuda") -> None:
        try:
            from external.HairFastGAN.hair_swap import HairFast, get_parser  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HairFastGAN is not available. Clone it into external/HairFastGAN "
                "or install it as a package."
            ) from exc
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

        args = get_parser().parse_args([])
        args.device = device
        self.model = HairFast(args)

    def transfer(
        self,
        face_img: np.ndarray,
        shape_img: np.ndarray,
        color_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if color_img is None:
            color_img = shape_img
        return self.model(face_img, shape_img, color_img)

