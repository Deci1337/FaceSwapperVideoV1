"""
Stable Diffusion Inpainting Pipeline Loader

Loads SD 1.5 Inpaint with IP-Adapter (FaceID) and ControlNet (OpenPose).
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model IDs from HuggingFace
SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
CONTROLNET_OPENPOSE = "lllyasviel/control_v11p_sd15_openpose"
IP_ADAPTER_FACEID = "h94/IP-Adapter-FaceID"


class SDInpaintLoader:
    """
    Manages Stable Diffusion Inpainting pipeline with ControlNet and IP-Adapter.
    """
    
    def __init__(
        self,
        cache_dir: Path,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded models
            device: 'cuda' or 'cpu'
            use_fp16: Use half precision (saves VRAM)
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.dtype = torch.float16 if use_fp16 and device == 'cuda' else torch.float32
        
        self.pipe = None
        self.controlnet = None
        self.ip_adapter_loaded = False
        
    def load(self) -> None:
        """
        Load all models. Call once before generating.
        """
        logger.info("Loading Stable Diffusion Inpainting pipeline...")
        
        try:
            from diffusers import (
                StableDiffusionInpaintPipeline,
                ControlNetModel,
                StableDiffusionControlNetInpaintPipeline
            )
        except ImportError:
            raise ImportError(
                "diffusers not installed. Run: pip install -r requirements-hair.txt"
            )
        
        # Load ControlNet for pose
        logger.info(f"Loading ControlNet: {CONTROLNET_OPENPOSE}")
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_OPENPOSE,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir
        )
        
        # Load SD Inpaint with ControlNet
        logger.info(f"Loading SD Inpaint: {SD_INPAINT_MODEL}")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL,
            controlnet=self.controlnet,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
            safety_checker=None,  # Disable for faster inference
            requires_safety_checker=False
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == 'cuda':
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"xformers not available: {e}")
                try:
                    self.pipe.enable_attention_slicing()
                    logger.info("Attention slicing enabled (fallback)")
                except:
                    pass
        
        # Load IP-Adapter for face identity
        self._load_ip_adapter()
        
        logger.info("SD pipeline loaded successfully")
        
    def _load_ip_adapter(self) -> None:
        """
        Load IP-Adapter FaceID for preserving face identity.
        """
        try:
            logger.info("Loading IP-Adapter FaceID...")
            
            # IP-Adapter FaceID requires insightface for face embedding
            # We already have this in the project
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                subfolder=None,
                weight_name="ip-adapter-faceid_sd15.bin"
            )
            
            # Set IP-Adapter scale (how much to use face identity)
            self.pipe.set_ip_adapter_scale(0.7)
            
            self.ip_adapter_loaded = True
            logger.info("IP-Adapter FaceID loaded")
            
        except Exception as e:
            logger.warning(f"IP-Adapter failed to load: {e}")
            logger.warning("Proceeding without face identity preservation")
            self.ip_adapter_loaded = False
    
    def set_identity_strength(self, strength: float) -> None:
        """
        Set how much to preserve face identity from reference.
        
        Args:
            strength: 0.0-1.0 (0=ignore reference face, 1=exact match)
        """
        if self.ip_adapter_loaded:
            self.pipe.set_ip_adapter_scale(strength)
            
    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        control_image: Image.Image,
        face_image: Optional[Image.Image] = None,
        prompt: str = "photorealistic woman with long hair",
        negative_prompt: str = "short hair, bald, blurry",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.85,
        controlnet_scale: float = 0.9,
        seed: int = 42,
        prev_latents: Optional[torch.Tensor] = None,
        latent_blend: float = 0.0
    ) -> Tuple[Image.Image, Optional[torch.Tensor]]:
        """
        Generate inpainted image with controlled pose and face identity.
        
        Args:
            image: Original frame (PIL Image)
            mask: Inpaint mask (white = inpaint region)
            control_image: Pose/depth control image
            face_image: Reference face for IP-Adapter
            prompt: Generation prompt
            negative_prompt: What to avoid
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence
            strength: How much to change masked region
            controlnet_scale: Pose adherence
            seed: Random seed for reproducibility
            prev_latents: Previous frame latents for consistency
            latent_blend: How much to blend with prev_latents
            
        Returns:
            (generated_image, latents) tuple
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        # Set generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Prepare kwargs
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "control_image": control_image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "controlnet_conditioning_scale": controlnet_scale,
            "generator": generator,
            "output_type": "pil"
        }
        
        # Add IP-Adapter face image if available
        if self.ip_adapter_loaded and face_image is not None:
            kwargs["ip_adapter_image"] = face_image
        
        # Run inference
        result = self.pipe(**kwargs)
        
        return result.images[0], None  # Latents extraction needs custom callback
    
    def unload(self) -> None:
        """
        Unload models to free VRAM.
        """
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("SD pipeline unloaded")

