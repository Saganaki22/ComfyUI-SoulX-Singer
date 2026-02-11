"""SoulX-Singer Model Loader Node."""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add SoulX-Singer to path
current_dir = Path(__file__).parent.parent
soulx_dir = current_dir / "SoulX-Singer"
if str(soulx_dir) not in sys.path:
    sys.path.insert(0, str(soulx_dir))

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger import SoulXSinger

# Import ComfyUI progress bar
try:
    from comfy.utils import ProgressBar
    COMFYUI_PROGRESS_AVAILABLE = True
except ImportError:
    COMFYUI_PROGRESS_AVAILABLE = False

logger = logging.getLogger("SoulX-Singer")

# Global model cache
_cached_model: Optional[Dict[str, Any]] = None
_cache_config: Optional[Dict[str, Any]] = None

# Get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = Path(folder_paths.models_dir) / "SoulX-Singer"
except Exception:
    MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "SoulX-Singer"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_available_models():
    """Scan for available model files and show download options for missing ones."""
    models = []
    seen = set()
    
    # Define expected models
    expected_models = {
        "SoulX-Singer_model_bf16": "SoulX-Singer_model_bf16.safetensors",
        "SoulX-Singer_model_fp32": "SoulX-Singer_model_fp32.safetensors",
    }
    
    if MODELS_DIR.exists():
        # Check which models exist locally
        for model_name, filename in expected_models.items():
            model_path = MODELS_DIR / filename
            if model_path.exists() or model_path.is_symlink():
                models.append(model_name)
                seen.add(model_name)
        
        # Also scan for any other .safetensors or .pt files
        for file in MODELS_DIR.glob("*.safetensors"):
            if file.is_file() or file.is_symlink():
                model_name = file.stem
                if model_name not in seen:
                    models.append(model_name)
                    seen.add(model_name)
        
        for file in MODELS_DIR.glob("*.pt"):
            if file.is_file() or file.is_symlink():
                model_name = file.stem
                if model_name not in seen:
                    models.append(model_name)
                    seen.add(model_name)
    
    # Add download options for missing models
    for model_name, filename in expected_models.items():
        if model_name not in seen:
            models.append(f"{model_name} (download)")
    
    if not models:
        models = ["SoulX-Singer_model_bf16 (download)"]
    
    return sorted(models)


def download_model_if_missing(model_name: str, include_fp32: bool = False):
    """Download model and all dependencies from HuggingFace if not present.
    
    Args:
        model_name: Name of the model to download (without .safetensors extension)
        include_fp32: If True, also download the fp32 model (SoulX-Singer_model.safetensors)
    """
    # Strip (download) suffix if present
    clean_model_name = model_name.replace(" (download)", "")
    model_path = MODELS_DIR / f"{clean_model_name}.safetensors"
    config_path = MODELS_DIR / "config.yaml"
    preprocessors_dir = MODELS_DIR / "preprocessors"
    
    # Check what needs to be downloaded
    needs_model = not model_path.exists()
    needs_config = not config_path.exists()
    needs_preprocessors = not preprocessors_dir.exists() or not any(preprocessors_dir.iterdir())
    
    # Also check if fp32 is requested
    fp32_path = MODELS_DIR / "SoulX-Singer_model_fp32.safetensors"
    needs_fp32 = include_fp32 and not fp32_path.exists()
    
    if not needs_model and not needs_config and not needs_preprocessors and not needs_fp32:
        return True
    
    logger.info(f"Downloading from HuggingFace: drbaph/SoulX-Singer...")
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        import shutil
        
        # First, download the specific model file if needed
        if needs_model:
            filename = f"{clean_model_name}.safetensors"
            logger.info(f"Downloading {filename}...")
            hf_hub_download(
                repo_id="drbaph/SoulX-Singer",
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
                resume_download=True
            )
        
        # Download fp32 model if requested
        if needs_fp32:
            logger.info("Downloading fp32 model (SoulX-Singer_model.safetensors)...")
            hf_hub_download(
                repo_id="drbaph/SoulX-Singer",
                filename="SoulX-Singer_model.safetensors",
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
                resume_download=True
            )
        
        # Download config if needed
        if needs_config:
            logger.info("Downloading config.yaml...")
            hf_hub_download(
                repo_id="drbaph/SoulX-Singer",
                filename="config.yaml",
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False
            )
        
        # Download preprocessors folder if needed (this is the big one ~5GB)
        if needs_preprocessors:
            logger.info("Downloading preprocessors folder (~5GB, this may take a while)...")
            # Download the entire preprocessors folder
            snapshot_download(
                repo_id="drbaph/SoulX-Singer",
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
                resume_download=True,
                allow_patterns=["preprocessors/**"]
            )
        
        logger.info(f"Download complete!")
        logger.info(f"Model: {model_path.exists()}")
        logger.info(f"Config: {config_path.exists()}")
        logger.info(f"Preprocessors: {preprocessors_dir.exists()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("Please manually download from https://huggingface.co/drbaph/SoulX-Singer")
        logger.error(f"Place files in: {MODELS_DIR}")
        return False


class SoulXSingerModelLoader:
    """Load SoulX-Singer model with dtype and attention configuration."""
    
    def __init__(self):
        self.model = None
        self.config = None
    
    @classmethod
    def INPUT_TYPES(cls):
        available = get_available_models()
        # Default to bf16 model if available, otherwise first available
        default_model = "SoulX-Singer_model_bf16"
        if available:
            # Check if bf16 exists (with or without download suffix)
            bf16_options = [m for m in available if "bf16" in m]
            if bf16_options:
                default_model = bf16_options[0]
            else:
                default_model = available[0]
        
        return {
            "required": {
                "model_name": (available, {
                    "default": default_model,
                    "tooltip": "Select model to load. '(download)' suffix means it will be downloaded from HuggingFace.",
                }),
                "dtype": (["fp32", "bf16"], {
                    "default": "bf16",
                    "tooltip": "Model precision: fp32 (full), bf16 (bfloat16). Note: fp16 has known FFT issues and is disabled.",
                }),
                "attention_type": (["sdpa", "sageattention"], {
                    "default": "sdpa",
                    "tooltip": "Attention mechanism: sdpa (PyTorch - default), sageattention (requires sageattention package)",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory between runs. Set to False to free memory after each execution.",
                }),
            },
        }
    
    RETURN_TYPES = ("SOULX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SoulX-Singer"
    DESCRIPTION = "Load SoulX-Singer model from ComfyUI/models/SoulX-Singer/"
    
    def load_model(
        self,
        model_name: str,
        dtype: str,
        attention_type: str,
        keep_loaded: bool,
    ) -> Tuple[Dict[str, Any]]:
        """Load the SoulX-Singer model.
        
        Args:
            model_name: Name of model file (without extension)
            dtype: Precision type (fp32/fp16/bf16)
            attention_type: Attention implementation
            keep_loaded: Whether to cache model between runs
            
        Returns:
            Tuple containing model dict with model, config, dtype, device
        """
        global _cached_model, _cache_config
        
        try:
            # Strip (download) suffix for cache key
            clean_model_name = model_name.replace(" (download)", "")
            
            # Check cache configuration
            current_config = {
                "model_name": clean_model_name,
                "dtype": dtype,
                "attention_type": attention_type,
            }
            
            # Use cached model only if keep_loaded is True AND config matches
            if keep_loaded and _cached_model is not None and _cache_config == current_config:
                logger.info("Using cached model")
                return (_cached_model,)
            
            # Clear old cache if:
            # 1. keep_loaded is False, OR
            # 2. Config changed (dtype or attention_type changed)
            if _cached_model is not None:
                if not keep_loaded:
                    logger.info("Clearing model from cache (keep_loaded=False)")
                elif _cache_config != current_config:
                    logger.info("Clearing model from cache (dtype or attention changed)")
                
                del _cached_model
                _cached_model = None
                _cache_config = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Check if user selected the fp32 model
            include_fp32 = ("SoulX-Singer_model_fp32" == clean_model_name)
            
            # Setup progress bar (3 stages: download/load weights/apply settings)
            total_steps = 3
            pbar = ProgressBar(total_steps) if COMFYUI_PROGRESS_AVAILABLE else None
            if pbar:
                logger.info(f"Progress: 0/{total_steps} - Starting model load...")
            
            # Stage 1: Download if missing
            # If user selected fp32 model, download it along with bf16+preprocessors
            # Otherwise just download bf16+preprocessors
            download_model_if_missing(clean_model_name, include_fp32=include_fp32)
            if pbar:
                pbar.update_absolute(1, total_steps)
            
            # Stage 2: Find and load model file (resolve symlinks)
            model_path = None
            for ext in ['.safetensors', '.pt']:
                path = MODELS_DIR / f"{clean_model_name}{ext}"
                # Check both regular files and symlinks
                if path.exists() or path.is_symlink():
                    # Resolve symlink to actual file
                    try:
                        resolved_path = path.resolve()
                        if resolved_path.exists():
                            model_path = resolved_path
                            logger.info(f"Found model: {path}")
                            if path.is_symlink():
                                logger.info(f"  (symlink resolved to: {resolved_path})")
                            break
                    except Exception as e:
                        logger.warning(f"Could not resolve path {path}: {e}")
                        continue
            
            if not model_path:
                raise FileNotFoundError(
                    f"Model not found: {clean_model_name}\n"
                    f"Expected location: {MODELS_DIR}\n"
                    f"Checked extensions: .safetensors, .pt\n"
                    f"Please download from https://huggingface.co/drbaph/SoulX-Singer"
                )
            
            # Load config
            config_path = MODELS_DIR / "config.yaml"
            if not config_path.exists():
                # Use bundled config
                config_path = soulx_dir / "soulxsinger" / "config" / "soulxsinger.yaml"
            
            config = load_config(config_path)
            logger.info(f"Config loaded from {config_path}")
            
            # Determine device and dtype
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if dtype == "fp16":
                torch_dtype = torch.float16
            elif dtype == "bf16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            logger.info(f"Loading model: {model_path}")
            logger.info(f"Device: {device}, Dtype: {dtype}")
            
            # Initialize model with attention type
            model = SoulXSinger(config, attention_type=attention_type)
            
            # Load weights
            if model_path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(str(model_path))
            else:
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=True)
            logger.info("Model weights loaded successfully")
            if pbar:
                pbar.update_absolute(2, total_steps)
            
            # Stage 3: Convert dtype and apply optimizations
            model = model.to(dtype=torch_dtype)
            model = model.to(device)
            model.eval()
            
            # Apply attention optimization
            if attention_type == "sageattention":
                logger.info("Attempting to apply SageAttention...")
                try:
                    # Try to import and apply SageAttention
                    import sageattention
                    from transformers.models.llama.modeling_llama import LlamaAttention
                    import torch.nn.functional as F
                    
                    # rotate_half helper function
                    def rotate_half(x):
                        """Rotates half the hidden dims of the input."""
                        x1 = x[..., : x.shape[-1] // 2]
                        x2 = x[..., x.shape[-1] // 2 :]
                        return torch.cat((-x2, x1), dim=-1)
                    
                    # Store original forward
                    _original_llama_attn_forward = LlamaAttention.forward
                    
                    def sage_llama_attention_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, position_embeddings=None, **kwargs):
                        """Replace SDPA with SageAttention in LlamaAttention."""
                        bsz, q_len, _ = hidden_states.size()
                        
                        # Get config values (handles different transformers versions)
                        num_heads = getattr(self, 'num_heads', getattr(self.config, 'num_attention_heads', 16))
                        num_key_value_heads = getattr(self, 'num_key_value_heads', getattr(self.config, 'num_key_value_heads', num_heads))
                        head_dim = getattr(self, 'head_dim', getattr(self.config, 'hidden_size', 1024) // num_heads)
                        hidden_size = getattr(self, 'hidden_size', getattr(self.config, 'hidden_size', 1024))
                        
                        query_states = self.q_proj(hidden_states)
                        key_states = self.k_proj(hidden_states)
                        value_states = self.v_proj(hidden_states)
                        
                        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
                        
                        # Apply rotary embeddings - position_embeddings already contains (cos, sin) from parent
                        if position_embeddings is not None:
                            cos, sin = position_embeddings
                            # Apply rotary embeddings manually
                            # Reshape cos/sin to match query_states shape: (batch, heads, seq_len, head_dim)
                            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
                            sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
                            
                            # Apply rotation
                            q_embed = (query_states * cos) + (rotate_half(query_states) * sin)
                            k_embed = (key_states * cos) + (rotate_half(key_states) * sin)
                            query_states, key_states = q_embed, k_embed
                        
                        # Use SageAttention with HND tensor layout (batch, heads, seq, dim)
                        attn_output = sageattention.sageattn(
                            query_states, key_states, value_states, 
                            tensor_layout='HND', 
                            is_causal=False
                        )
                        
                        attn_output = attn_output.transpose(1, 2).contiguous()
                        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
                        attn_output = self.o_proj(attn_output)
                        
                        if output_attentions:
                            return attn_output, None, past_key_value
                        return attn_output, past_key_value
                    
                    # Monkey-patch the attention forward
                    LlamaAttention.forward = sage_llama_attention_forward
                    logger.info("SageAttention successfully applied to LlamaAttention layers")
                    
                except ImportError:
                    logger.warning("SageAttention not installed. Falling back to SDPA.")
                    logger.warning("To use SageAttention, install with: pip install sageattention")
                except Exception as e:
                    logger.warning(f"Failed to apply SageAttention: {e}")
                    logger.warning("Falling back to SDPA")
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
            
            if pbar:
                pbar.update_absolute(3, total_steps)
            
            # Create model info dict
            model_info = {
                "model": model,
                "config": config,
                "dtype": torch_dtype,
                "device": device,
                "attention_type": attention_type,
                "keep_loaded": keep_loaded,
            }
            
            # Cache if keep_loaded is True
            if keep_loaded:
                _cached_model = model_info
                _cache_config = current_config
                logger.info("Model cached for future use")
            
            return (model_info,)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force reload if parameters change."""
        return hash(str(kwargs))
