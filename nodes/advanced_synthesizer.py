"""SoulX-Singer Advanced Synthesizer Node with pre-processed metadata support."""

import os
import sys
import torch
import logging
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

# Add SoulX-Singer to path
current_dir = Path(__file__).parent.parent
soulx_dir = current_dir / "SoulX-Singer"
if str(soulx_dir) not in sys.path:
    sys.path.insert(0, str(soulx_dir))

from soulxsinger.utils.data_processor import DataProcessor

# Import ComfyUI progress bar
try:
    from comfy.utils import ProgressBar
    COMFYUI_PROGRESS_AVAILABLE = True
except ImportError:
    COMFYUI_PROGRESS_AVAILABLE = False

# Import interruption support
try:
    import comfy.model_management as mm
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False

logger = logging.getLogger("SoulX-Singer")


def format_audio_for_comfyui(audio: np.ndarray, sample_rate: int = 24000) -> Dict[str, Any]:
    """Format audio tensor for ComfyUI."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    audio = audio.contiguous().cpu().float()
    
    if audio.device.type == "cuda":
        torch.cuda.synchronize()
    
    return {
        "waveform": audio,
        "sample_rate": sample_rate,
    }


class SoulXSingerAdvanced:
    """Advanced synthesizer using pre-processed metadata files (for MIDI editor workflow)."""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SOULX_MODEL", {
                    "tooltip": "Model from SoulX-Singer Model Loader",
                }),
                "prompt_audio": ("AUDIO", {
                    "tooltip": "Reference singing voice audio",
                }),
                "prompt_metadata_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to prompt metadata JSON file (from preprocessing or MIDI editor)",
                }),
                "target_metadata_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to target metadata JSON file (from preprocessing or MIDI editor)",
                }),
                "control_mode": (["melody", "score"], {
                    "default": "melody",
                    "tooltip": "melody: F0 contour control, score: MIDI note control",
                }),
                "auto_pitch_shift": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically match pitch range between prompt and target",
                }),
                "pitch_shift": ("INT", {
                    "default": 0,
                    "min": -12,
                    "max": 12,
                    "step": 1,
                    "tooltip": "Manual pitch shift in semitones (ignored if auto_pitch_shift is enabled)",
                }),
                "n_steps": ("INT", {
                    "default": 32,
                    "min": 16,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Diffusion steps (higher = better quality but slower)",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale (higher = follows prompt more strictly)",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "SoulX-Singer"
    DESCRIPTION = "Advanced singing voice synthesis using pre-processed metadata files"
    
    def _check_interrupt(self):
        """Check if user requested cancellation."""
        if INTERRUPTION_SUPPORT:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise
    
    def synthesize(
        self,
        model: Dict[str, Any],
        prompt_audio: Dict[str, Any],
        prompt_metadata_path: str,
        target_metadata_path: str,
        control_mode: str,
        auto_pitch_shift: bool,
        pitch_shift: int,
        n_steps: int,
        cfg_scale: float,
    ) -> Tuple[Dict[str, Any]]:
        """Synthesize singing voice using pre-processed metadata.
        
        Args:
            model: Model dict from loader
            prompt_audio: Reference audio (ComfyUI AUDIO format)
            prompt_metadata_path: Path to prompt metadata JSON
            target_metadata_path: Path to target metadata JSON
            control_mode: melody or score
            auto_pitch_shift: Auto-match pitch ranges
            pitch_shift: Manual pitch shift (semitones)
            n_steps: Diffusion steps
            cfg_scale: CFG guidance scale
            
        Returns:
            Tuple with generated audio in ComfyUI format
        """
        try:
            # Validate paths
            if not prompt_metadata_path or not Path(prompt_metadata_path).exists():
                raise FileNotFoundError(f"Prompt metadata not found: {prompt_metadata_path}")
            
            if not target_metadata_path or not Path(target_metadata_path).exists():
                raise FileNotFoundError(f"Target metadata not found: {target_metadata_path}")
            
            # Setup progress bar
            # Stages: load metadata (1), process (1), synthesize (1)
            total_steps = 3
            pbar = ProgressBar(total_steps) if COMFYUI_PROGRESS_AVAILABLE else None
            
            logger.info("=" * 60)
            logger.info("SoulX-Singer Advanced Synthesis (Metadata Mode)")
            logger.info("=" * 60)
            
            self._check_interrupt()
            
            # Stage 1: Load metadata files
            logger.info("Stage 1/3: Loading metadata files...")
            
            with open(prompt_metadata_path, "r", encoding="utf-8") as f:
                prompt_meta_list = json.load(f)
            
            if not prompt_meta_list:
                raise ValueError("Prompt metadata is empty")
            
            prompt_meta = prompt_meta_list[0]  # Use first segment
            
            with open(target_metadata_path, "r", encoding="utf-8") as f:
                target_meta_list = json.load(f)
            
            if not target_meta_list:
                raise ValueError("Target metadata is empty")
            
            logger.info(f"Loaded {len(target_meta_list)} target segments")
            
            if pbar:
                pbar.update_absolute(1, total_steps)
            
            self._check_interrupt()
            
            # Stage 2: Process metadata with DataProcessor
            logger.info("Stage 2/3: Processing metadata...")
            
            config = model["config"]
            device = model["device"]
            
            phoneset_path = soulx_dir / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
            
            data_processor = DataProcessor(
                hop_size=config.audio.hop_size,
                sample_rate=config.audio.sample_rate,
                phoneset_path=str(phoneset_path),
                device=device,
            )
            
            # Convert prompt audio to wav for processing
            waveform = prompt_audio["waveform"]  # [batch, channels, samples]
            sample_rate = prompt_audio["sample_rate"]
            
            # Save to temp file
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                prompt_wav_path = tmp.name
                audio_data = waveform[0].cpu().numpy().T
                if audio_data.shape[1] == 1:
                    audio_data = audio_data.squeeze()
                sf.write(prompt_wav_path, audio_data, sample_rate)
            
            # Process prompt
            infer_prompt_data = data_processor.process(prompt_meta, prompt_wav_path)
            
            # Clean up temp file
            try:
                os.unlink(prompt_wav_path)
            except Exception:
                pass
            
            if pbar:
                pbar.update_absolute(2, total_steps)
            
            self._check_interrupt()
            
            # Stage 3: Synthesize
            logger.info("Stage 3/3: Synthesizing singing voice...")
            logger.info(f"Control mode: {control_mode}")
            logger.info(f"Pitch shift: {'auto' if auto_pitch_shift else pitch_shift}")
            logger.info(f"Steps: {n_steps}, CFG: {cfg_scale}")
            
            # Synthesize all target segments
            generated_len = int(target_meta_list[-1]["time"][1] / 1000 * config.audio.sample_rate)
            generated_merged = np.zeros(generated_len, dtype=np.float32)
            
            for idx, target_meta in enumerate(target_meta_list):
                self._check_interrupt()
                
                logger.info(f"Processing segment {idx + 1}/{len(target_meta_list)}")
                
                start_sample = int(target_meta["time"][0] / 1000 * config.audio.sample_rate)
                infer_target_data = data_processor.process(target_meta, None)
                
                infer_data = {
                    "prompt": infer_prompt_data,
                    "target": infer_target_data,
                }
                
                # Run inference
                with torch.no_grad():
                    generated_audio = model["model"].infer(
                        infer_data,
                        auto_shift=auto_pitch_shift,
                        pitch_shift=pitch_shift,
                        n_steps=n_steps,
                        cfg=cfg_scale,
                        control=control_mode,
                    )
                
                # Merge segment
                generated_audio = generated_audio.squeeze().cpu().numpy()
                generated_merged[start_sample : start_sample + generated_audio.shape[0]] = generated_audio
            
            if pbar:
                pbar.update_absolute(3, total_steps)
            
            logger.info("Synthesis complete!")
            logger.info("=" * 60)
            
            # Format output for ComfyUI
            output_audio = format_audio_for_comfyui(generated_merged, 24000)
            
            return (output_audio,)
            
        except Exception as e:
            # Handle interruption
            if INTERRUPTION_SUPPORT:
                from comfy.model_management import InterruptProcessingException
                if isinstance(e, InterruptProcessingException):
                    logger.info("Synthesis interrupted by user")
                    raise
            
            logger.error(f"Synthesis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution on input changes."""
        return hash(str(kwargs))
