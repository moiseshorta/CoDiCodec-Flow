"""CoreML conversion and inference utilities for FlowDiT.

This module provides utilities to convert FlowDiT models to CoreML format
and run inference with CoreML as an optional backend (with fallback to MPS).

CoreML conversion is optional and requires coremltools to be installed.
The conversion process:
1. Load the PyTorch model
2. Convert to TorchScript via tracing
3. Convert TorchScript to CoreML using coremltools
4. Save as .mlpackage file

Note: CoreML has limitations for dynamic shapes and complex control flow.
For realtime audio generation with sliding windows, MPS (PyTorch) is recommended.
CoreML is provided for experimentation and potential deployment scenarios.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False

from .config import ModelConfig
from .model.dit import FlowDiT
from .model.ema import EMA
from .utils import get_logger

logger = get_logger("flow.coreml_utils")


def check_coreml_available() -> bool:
    """Check if coremltools is available."""
    return COREML_AVAILABLE


def load_model_for_conversion(
    ckpt_path: str,
    *,
    use_ema: bool = True,
) -> FlowDiT:
    """Load a FlowDiT model for CoreML conversion.
    
    This mirrors the load_model function in realtime.py but returns
    the model in eval mode on CPU (required for CoreML conversion).
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        use_ema: Whether to load EMA weights (recommended)
        
    Returns:
        FlowDiT model in eval mode on CPU
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in sd or "model" not in sd["config"]:
        raise RuntimeError(
            f"checkpoint {ckpt_path} has no 'config.model' field; cannot rebuild architecture"
        )
    model_cfg = ModelConfig(**sd["config"]["model"])
    model = FlowDiT(
        latent_dim=model_cfg.latent_dim,
        block_size=model_cfg.block_size,
        dim=model_cfg.dim,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        head_dim=model_cfg.head_dim,
        mlp_mult=model_cfg.mlp_mult,
        cond_dim=model_cfg.cond_dim,
        max_seq_len=model_cfg.max_seq_len,
        dropout=0.0,  # no dropout at inference
    )
    if use_ema:
        ema_state = sd.get("ema")
        if ema_state is None:
            raise RuntimeError(f"--use-ema set but {ckpt_path} has no 'ema' field")
        ema = EMA(model, decay=0.0)
        ema.load_state_dict(ema_state)
        ema.copy_to(model)
        logger.info("Loaded EMA weights from %s", ckpt_path)
    else:
        if "model" not in sd:
            raise RuntimeError(f"checkpoint {ckpt_path} has no 'model' field")
        model.load_state_dict(sd["model"])
        logger.info("Loaded raw (non-EMA) weights from %s", ckpt_path)
    model.eval()
    return model


def convert_to_coreml(
    model: FlowDiT,
    output_path: str,
    *,
    context_chunks: int = 32,
    latent_dim: int = 128,
    block_size: int = 48,
    min_deployment_target: str = "macos13",
) -> Optional[ct.models.MLModel]:
    """Convert a FlowDiT model to CoreML format.
    
    Args:
        model: FlowDiT model in eval mode (should be on CPU)
        output_path: Path to save the CoreML model (.mlpackage)
        context_chunks: Number of context chunks for tracing (affects input shape)
        latent_dim: Latent dimension
        block_size: Block size (tokens per chunk)
        min_deployment_target: Minimum deployment target (e.g., "macos13", "ios16")
        
    Returns:
        CoreML model if successful, None otherwise
    """
    if not COREML_AVAILABLE:
        logger.error("coremltools is not installed. Install with: pip install coremltools")
        return None
    
    # Map string target to coremltools.target enum
    target_map = {
        "macos13": ct.target.macOS13,
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
        "macos14": ct.target.macOS14,
    }
    if min_deployment_target not in target_map:
        logger.error("Invalid deployment target: %s. Valid options: %s", 
                    min_deployment_target, list(target_map.keys()))
        return None
    
    target = target_map[min_deployment_target]
    
    logger.info("Converting model to CoreML...")
    
    # Calculate input shape: [batch=1, seq_len, latent_dim]
    seq_len = context_chunks * block_size
    
    # Create example input for tracing
    # FlowDiT expects: x [B, L, D], t [B, L], attn_mask [L, L]
    example_x = torch.randn(1, seq_len, latent_dim)
    example_t = torch.rand(1, seq_len)
    example_attn_mask = torch.ones(seq_len, seq_len)
    
    try:
        # Trace the model
        logger.info("Tracing model with example input shape: %s", example_x.shape)
        traced_model = torch.jit.trace(
            model,
            (example_x, example_t, example_attn_mask),
        )
        
        # Verify traced model works
        with torch.no_grad():
            output = traced_model(example_x, example_t, example_attn_mask)
            logger.info("Traced model output shape: %s", output.shape)
        
        # Convert to CoreML
        logger.info("Converting TorchScript to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType(name="x", shape=example_x.shape, dtype=np.float32),
                ct.TensorType(name="t", shape=example_t.shape, dtype=np.float32),
                ct.TensorType(name="attn_mask", shape=example_attn_mask.shape, dtype=np.float32),
            ],
            minimum_deployment_target=target,
        )
        
        # Save the model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))
        logger.info("CoreML model saved to: %s", output_path)
        
        return mlmodel
        
    except Exception as e:
        logger.error("Failed to convert model to CoreML: %s", e)
        logger.warning("CoreML conversion failed. This is expected for models with dynamic shapes or complex control flow.")
        logger.info("Falling back to PyTorch MPS backend.")
        return None


def convert_checkpoint_to_coreml(
    ckpt_path: str,
    output_path: str,
    *,
    use_ema: bool = True,
    context_chunks: int = 32,
    min_deployment_target: str = "macos13",
) -> bool:
    """Convert a checkpoint to CoreML format.
    
    This is a convenience function that loads the model and converts it.
    
    Args:
        ckpt_path: Path to checkpoint file (.pt)
        output_path: Path to save the CoreML model (.mlpackage)
        use_ema: Whether to load EMA weights
        context_chunks: Number of context chunks for tracing
        min_deployment_target: Minimum deployment target
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    if not COREML_AVAILABLE:
        logger.error("coremltools is not installed. Install with: pip install coremltools")
        return False
    
    try:
        model = load_model_for_conversion(ckpt_path, use_ema=use_ema)
        mlmodel = convert_to_coreml(
            model,
            output_path,
            context_chunks=context_chunks,
            latent_dim=model.latent_dim,
            block_size=model.block_size,
            min_deployment_target=min_deployment_target,
        )
        return mlmodel is not None
    except Exception as e:
        logger.error("Failed to convert checkpoint %s: %s", ckpt_path, e)
        return False


class CoreMLBackend:
    """CoreML inference backend with fallback to PyTorch.
    
    This class provides a CoreML inference path that falls back to PyTorch
    if CoreML is not available or fails to load the model.
    """
    
    def __init__(
        self,
        coreml_path: Optional[str] = None,
        pytorch_model: Optional[FlowDiT] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the CoreML backend.
        
        Args:
            coreml_path: Path to CoreML model (.mlpackage). If None, uses PyTorch only.
            pytorch_model: Fallback PyTorch model. Required if coreml_path is None or fails.
            device: Device for PyTorch fallback
        """
        self.coreml_path = coreml_path
        self.pytorch_model = pytorch_model
        self.device = device
        self.use_coreml = False
        self.coreml_model = None
        
        if coreml_path and COREML_AVAILABLE and Path(coreml_path).exists():
            try:
                logger.info("Loading CoreML model from: %s", coreml_path)
                self.coreml_model = ct.models.MLModel(coreml_path)
                self.use_coreml = True
                logger.info("CoreML model loaded successfully")
            except Exception as e:
                logger.warning("Failed to load CoreML model: %s", e)
                logger.info("Falling back to PyTorch backend")
        
        if not self.use_coreml and pytorch_model is None:
            raise ValueError("Either coreml_path must be valid or pytorch_model must be provided")
    
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference.
        
        Args:
            x: Input tensor [B, L, D]
            t: Time tensor [B, L]
            attn_mask: Attention mask [L, L]
            
        Returns:
            Output tensor [B, L, D]
        """
        if self.use_coreml and self.coreml_model is not None:
            try:
                # CoreML inference
                # Convert tensors to numpy
                x_np = x.cpu().numpy().astype(np.float32)
                t_np = t.cpu().numpy().astype(np.float32)
                attn_mask_np = attn_mask.cpu().numpy().astype(np.float32)
                
                # Run CoreML prediction
                output_dict = self.coreml_model.predict({
                    "x": x_np,
                    "t": t_np,
                    "attn_mask": attn_mask_np,
                })
                
                # Get the output (the key may vary, get the first tensor output)
                output_keys = [k for k, v in output_dict.items() if isinstance(v, np.ndarray)]
                if not output_keys:
                    raise RuntimeError("No tensor outputs found in CoreML model")
                output_key = output_keys[0]
                
                # Convert back to torch tensor
                output = torch.from_numpy(output_dict[output_key]).to(self.device)
                return output
                
            except Exception as e:
                logger.warning("CoreML inference failed: %s", e)
                logger.info("Falling back to PyTorch for this call")
        
        # PyTorch fallback
        if self.pytorch_model is None:
            raise RuntimeError("PyTorch model not available for fallback")
        
        return self.pytorch_model(x, t, attn_mask=attn_mask)
    
    def is_using_coreml(self) -> bool:
        """Check if CoreML is being used."""
        return self.use_coreml
