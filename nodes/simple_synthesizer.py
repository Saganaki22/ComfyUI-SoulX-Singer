"""SoulX-Singer Simple Synthesizer Node with auto-preprocessing."""

import os
import sys

# Set NLTK data directory to be within our package folder, not roaming
nltk_data_dir = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
os.environ['NLTK_DATA'] = os.path.abspath(nltk_data_dir)

# Suppress NeMo verbose logging
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import logging
import tempfile
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

# Lazy load preprocessing pipeline
_preprocess_pipeline = None

# Get ComfyUI models directory
try:
    import folder_paths
    MODELS_DIR = Path(folder_paths.models_dir) / "SoulX-Singer"
except Exception:
    MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "SoulX-Singer"

# Preprocessing models directory - now in ComfyUI/models/SoulX-Singer/preprocessors
PREPROCESS_MODELS_DIR = MODELS_DIR / "preprocessors"


def download_preprocessing_models():
    """Download preprocessing models from HuggingFace if missing."""
    # Check if directory exists and contains files (including symlinked directories)
    try:
        if PREPROCESS_MODELS_DIR.exists() or PREPROCESS_MODELS_DIR.is_symlink():
            # Resolve symlink if it is one
            resolved_dir = PREPROCESS_MODELS_DIR.resolve()
            if resolved_dir.exists() and any(resolved_dir.iterdir()):
                logger.info(f"Preprocessing models found at {PREPROCESS_MODELS_DIR}")
                if PREPROCESS_MODELS_DIR.is_symlink():
                    logger.info(f"  (symlink resolved to: {resolved_dir})")
                return True
    except Exception as e:
        logger.warning(f"Could not check preprocessing models directory: {e}")
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING MODELS NOT FOUND")
    logger.info("=" * 60)
    logger.info("Downloading ~5GB of preprocessing models from HuggingFace...")
    logger.info("Repository: Soul-AILab/SoulX-Singer-Preprocess")
    logger.info("This is a one-time download.")
    logger.info("=" * 60)
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download all preprocessing models
        snapshot_download(
            repo_id="Soul-AILab/SoulX-Singer-Preprocess",
            local_dir=str(PREPROCESS_MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        
        logger.info("=" * 60)
        logger.info("Preprocessing models downloaded successfully!")
        logger.info(f"Location: {PREPROCESS_MODELS_DIR}")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FAILED TO DOWNLOAD PREPROCESSING MODELS")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Please manually download from:")
        logger.error("https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess")
        logger.error("")
        logger.error("Extract to:")
        logger.error(f"{PREPROCESS_MODELS_DIR}")
        logger.error("=" * 60)
        return False


def get_preprocess_pipeline(device: str = "cuda"):
    """Lazy load preprocessing pipeline on first use."""
    global _preprocess_pipeline
    
    if _preprocess_pipeline is None:
        logger.info("Initializing preprocessing pipeline (first time only)...")
        
        # Download models if needed
        if not download_preprocessing_models():
            raise RuntimeError(
                "Preprocessing models not available. "
                "Please download manually from HuggingFace: Soul-AILab/SoulX-Singer-Preprocess"
            )
        
        try:
            from preprocess.pipeline import PreprocessPipeline
            
            _preprocess_pipeline = PreprocessPipeline(
                device=device,
                language="Mandarin",  # Will be overridden per call
                save_dir=str(tempfile.gettempdir()),
                vocal_sep=True,
                max_merge_duration=60000,
            )
            logger.info("Preprocessing pipeline loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load preprocessing pipeline: {e}")
            raise
    
    return _preprocess_pipeline


def audio_to_wav_file(audio_dict: Dict[str, Any], output_path: str):
    """Convert ComfyUI AUDIO format to wav file.
    
    Args:
        audio_dict: Dict with 'waveform' (tensor) and 'sample_rate'
        output_path: Path to save wav file
    """
    import soundfile as sf
    
    waveform = audio_dict["waveform"]  # Shape: [batch, channels, samples]
    sample_rate = audio_dict["sample_rate"]
    
    # Convert to numpy and reshape for soundfile
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Get first batch, transpose to [samples, channels]
    audio_data = waveform[0].T
    
    # If mono, squeeze to 1D
    if audio_data.shape[1] == 1:
        audio_data = audio_data.squeeze()
    
    sf.write(output_path, audio_data, sample_rate)


def format_audio_for_comfyui(audio: np.ndarray, sample_rate: int = 24000) -> Dict[str, Any]:
    """Format audio tensor for ComfyUI.
    
    Args:
        audio: Audio numpy array
        sample_rate: Sample rate
        
    Returns:
        Dict with 'waveform' and 'sample_rate'
    """
    # Convert to torch tensor
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    
    # Ensure shape: [batch, channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    # Ensure contiguous for playback
    audio = audio.contiguous()
    
    # Move to CPU
    audio = audio.cpu().float()
    
    # Synchronize device
    if audio.device.type == "cuda":
        torch.cuda.synchronize()
    
    return {
        "waveform": audio,
        "sample_rate": sample_rate,
    }


class SoulXSingerSimple:
    """Simple synthesizer with auto-preprocessing and AUDIO noodle inputs."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "soulx_singer"
        self.temp_dir.mkdir(exist_ok=True)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SOULX_MODEL", {
                    "tooltip": "Model from SoulX-Singer Model Loader",
                }),
                "prompt_audio": ("AUDIO", {
                    "tooltip": "Reference singing voice (3-10 seconds recommended)",
                }),
                "target_audio": ("AUDIO", {
                    "tooltip": "Target melody/score to synthesize",
                }),
                "prompt_language": (["Mandarin", "English", "Cantonese"], {
                    "default": "Mandarin",
                    "tooltip": "Language of prompt audio lyrics",
                }),
                "target_language": (["Mandarin", "English", "Cantonese"], {
                    "default": "Mandarin",
                    "tooltip": "Language of target audio lyrics",
                }),
                "control_mode": (["melody", "score"], {
                    "default": "melody",
                    "tooltip": "melody: F0 contour control, score: MIDI note control",
                }),
                "vocal_sep_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply vocal separation to prompt audio",
                }),
                "vocal_sep_target": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply vocal separation to target audio",
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
    DESCRIPTION = "Simple singing voice synthesis with auto-preprocessing"
    
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
        target_audio: Dict[str, Any],
        prompt_language: str,
        target_language: str,
        control_mode: str,
        vocal_sep_prompt: bool,
        vocal_sep_target: bool,
        auto_pitch_shift: bool,
        pitch_shift: int,
        n_steps: int,
        cfg_scale: float,
    ) -> Tuple[Dict[str, Any]]:
        """Synthesize singing voice with auto-preprocessing.
        
        Args:
            model: Model dict from loader
            prompt_audio: Reference audio (ComfyUI AUDIO format)
            target_audio: Target audio (ComfyUI AUDIO format)
            prompt_language: Language of prompt
            target_language: Language of target
            control_mode: melody or score
            vocal_sep_prompt: Apply vocal separation to prompt
            vocal_sep_target: Apply vocal separation to target
            auto_pitch_shift: Auto-match pitch ranges
            pitch_shift: Manual pitch shift (semitones)
            n_steps: Diffusion steps
            cfg_scale: CFG guidance scale
            
        Returns:
            Tuple with generated audio in ComfyUI format
        """
        try:
            # Setup progress bar
            # Stages: save audio (1), preprocess prompt (1), preprocess target (1), synthesize (1)
            total_steps = 4
            pbar = ProgressBar(total_steps) if COMFYUI_PROGRESS_AVAILABLE else None
            
            logger.info("=" * 60)
            logger.info("SoulX-Singer Simple Synthesis")
            logger.info("=" * 60)
            
            self._check_interrupt()
            
            # Stage 1: Save audio to temp files
            logger.info("Stage 1/4: Preparing audio files...")
            prompt_path = self.temp_dir / "prompt.wav"
            target_path = self.temp_dir / "target.wav"
            
            audio_to_wav_file(prompt_audio, str(prompt_path))
            audio_to_wav_file(target_audio, str(target_path))
            
            if pbar:
                pbar.update_absolute(1, total_steps)
            
            self._check_interrupt()
            
            # Stage 2: Preprocess prompt
            logger.info("Stage 2/4: Preprocessing prompt audio...")
            device = model["device"]
            pipeline = get_preprocess_pipeline(device)
            
            # Update pipeline settings
            pipeline.language = prompt_language
            pipeline.vocal_sep = vocal_sep_prompt
            pipeline.save_dir = str(self.temp_dir / "prompt_metadata")
            
            # Run preprocessing
            pipeline.run(
                audio_path=str(prompt_path),
                vocal_sep=vocal_sep_prompt,
                max_merge_duration=20000,
                language=prompt_language,
            )
            
            # Load generated metadata
            prompt_meta_path = self.temp_dir / "prompt_metadata" / "metadata.json"
            with open(prompt_meta_path, "r", encoding="utf-8") as f:
                prompt_meta_list = json.load(f)
            
            if not prompt_meta_list:
                raise ValueError("Prompt preprocessing failed - no metadata generated")
            
            prompt_meta = prompt_meta_list[0]  # Use first segment
            
            if pbar:
                pbar.update_absolute(2, total_steps)
            
            self._check_interrupt()
            
            # Stage 3: Preprocess target
            logger.info("Stage 3/4: Preprocessing target audio...")
            pipeline.language = target_language
            pipeline.vocal_sep = vocal_sep_target
            pipeline.save_dir = str(self.temp_dir / "target_metadata")
            
            pipeline.run(
                audio_path=str(target_path),
                vocal_sep=vocal_sep_target,
                max_merge_duration=60000,
                language=target_language,
            )
            
            # Load generated metadata
            target_meta_path = self.temp_dir / "target_metadata" / "metadata.json"
            with open(target_meta_path, "r", encoding="utf-8") as f:
                target_meta_list = json.load(f)
            
            if not target_meta_list:
                raise ValueError("Target preprocessing failed - no metadata generated")
            
            if pbar:
                pbar.update_absolute(3, total_steps)
            
            self._check_interrupt()
            
            # Stage 4: Synthesize
            logger.info("Stage 4/4: Synthesizing singing voice...")
            logger.info(f"Control mode: {control_mode}")
            logger.info(f"Pitch shift: {'auto' if auto_pitch_shift else pitch_shift}")
            logger.info(f"Steps: {n_steps}, CFG: {cfg_scale}")
            
            # Process metadata with DataProcessor
            config = model["config"]
            phoneset_path = soulx_dir / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
            
            data_processor = DataProcessor(
                hop_size=config.audio.hop_size,
                sample_rate=config.audio.sample_rate,
                phoneset_path=str(phoneset_path),
                device=device,
            )
            
            # Process prompt
            infer_prompt_data = data_processor.process(prompt_meta, str(prompt_path))
            
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
                pbar.update_absolute(4, total_steps)
            
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
