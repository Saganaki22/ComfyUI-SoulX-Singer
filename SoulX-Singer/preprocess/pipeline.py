import json
import os
import shutil
import soundfile as sf
from pathlib import Path
import librosa

from preprocess.utils import convert_metadata, merge_short_segments

from preprocess.tools import (
    F0Extractor,
    VocalDetector,
    VocalSeparator,
    NoteTranscriber,
    LyricTranscriber,
)


class PreprocessPipeline:
    def __init__(self, device: str, language: str, save_dir: str, vocal_sep: bool = True, max_merge_duration: int = 60000,
                 f0_half_precision: bool = False, torch_compile: bool = False):
        self.device = device
        self.language = language
        self.save_dir = save_dir
        self.vocal_sep = vocal_sep
        self.max_merge_duration = max_merge_duration

        # Get absolute path to preprocessing models
        # Try multiple possible locations for better compatibility
        
        # Method 1: Try using folder_paths (ComfyUI integration)
        base_dir = None
        try:
            import folder_paths
            models_base = Path(folder_paths.models_dir) / "SoulX-Singer" / "preprocessors"
            if models_base.exists():
                base_dir = models_base
        except Exception:
            pass
        
        # Method 2: Navigate from this file's location
        # This file is at: custom_nodes/ComfyUI-SoulX-Singer/SoulX-Singer/preprocess/pipeline.py
        if base_dir is None:
            custom_nodes_dir = Path(__file__).parent.parent.parent.parent
            comfyui_dir = custom_nodes_dir.parent
            fallback_dir = comfyui_dir / "models" / "SoulX-Singer" / "preprocessors"
            if fallback_dir.exists():
                base_dir = fallback_dir
        
        # Method 3: Check if symlink or environment variable exists
        if base_dir is None:
            env_path = os.environ.get('SOULX_PREPROCESS_MODELS')
            if env_path and Path(env_path).exists():
                base_dir = Path(env_path)
        
        # Method 4: Check current working directory relative path
        if base_dir is None:
            cwd_relative = Path.cwd() / "models" / "SoulX-Singer" / "preprocessors"
            if cwd_relative.exists():
                base_dir = cwd_relative
        
        if base_dir is None or not base_dir.exists():
            raise FileNotFoundError(
                f"Preprocessing models not found!\n"
                f"Searched locations:\n"
                f"  1. ComfyUI/models/SoulX-Singer/preprocessors/\n"
                f"  2. Environment variable: SOULX_PREPROCESS_MODELS\n"
                f"  3. Current working directory: {Path.cwd() / 'models' / 'SoulX-Singer' / 'preprocessors'}\n"
                f"\n"
                f"Please ensure preprocessing models are downloaded to one of these locations.\n"
                f"Or set SOULX_PREPROCESS_MODELS environment variable to the correct path."
            )
        
        # Resolve symlinks to actual path
        base_dir = base_dir.resolve()
        
        if vocal_sep:
            self.vocal_separator = VocalSeparator(
                sep_model_path=str(base_dir / "mel-band-roformer-karaoke" / "mel_band_roformer_karaoke_becruily.ckpt"),
                sep_config_path=str(base_dir / "mel-band-roformer-karaoke" / "config_karaoke_becruily.yaml"),
                der_model_path=str(base_dir / "dereverb_mel_band_roformer" / "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
                der_config_path=str(base_dir / "dereverb_mel_band_roformer" / "dereverb_mel_band_roformer_anvuew.yaml"),
                device=device
            )
        else:
            self.vocal_separator = None
        self.f0_extractor = F0Extractor(
            model_path=str(base_dir / "rmvpe" / "rmvpe.pt"),
            device=device,
            is_half=f0_half_precision,
        )
        if torch_compile:
            import torch
            try:
                self.f0_extractor.model.model = torch.compile(self.f0_extractor.model.model)
                print("[preprocess] torch.compile applied to RMVPE model")
            except Exception as e:
                print(f"[preprocess] torch.compile failed (falling back to eager): {e}")
        self.vocal_detector = VocalDetector(
            cut_wavs_output_dir=  f"{save_dir}/cut_wavs",
        )
        self.lyric_transcriber = LyricTranscriber(
            zh_model_path=str(base_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
            en_model_path=str(base_dir / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo"),
            device=device
        )
        self.note_transcriber = NoteTranscriber(
            rosvot_model_path=str(base_dir / "rosvot" / "rosvot" / "model.pt"), 
            rwbd_model_path=str(base_dir / "rosvot" / "rwbd" / "model.pt"), 
            device=device
        )

    def run(
        self,
        audio_path: str,
        vocal_sep: bool = True,
        max_merge_duration: int = 60000,
        language: str = "Mandarin"
    ) -> None:
        vocal_sep = self.vocal_sep if vocal_sep is None else vocal_sep
        max_merge_duration = self.max_merge_duration if max_merge_duration is None else max_merge_duration
        language = self.language if language is None else language
        output_dir = Path(self.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if vocal_sep:
            # Perform vocal/accompaniment separation
            sep = self.vocal_separator.process(audio_path)
            vocal = sep.vocals_dereverbed.T
            acc = sep.accompaniment.T
            sample_rate = sep.sample_rate

            vocal_path = output_dir / "vocal.wav"
            acc_path = output_dir / "acc.wav"
            sf.write(vocal_path, vocal, sample_rate)
            sf.write(acc_path, acc, sample_rate)
        else:
            # Use the original audio as vocal source (no separation)
            vocal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            vocal_path = output_dir / "vocal.wav"
            sf.write(vocal_path, vocal, sample_rate)

        vocal_f0 = self.f0_extractor.process(str(vocal_path))
        segments = self.vocal_detector.process(str(vocal_path), f0=vocal_f0)

        metadata = []
        for seg in segments:
            self.f0_extractor.process(seg["wav_fn"], f0_path=seg["wav_fn"].replace(".wav", "_f0.npy"))
            words, durs = self.lyric_transcriber.process(
                seg["wav_fn"], language
            )
            seg["words"] = words
            seg["word_durs"] = durs
            seg["language"] = language
            metadata.append(
                self.note_transcriber.process(seg, segment_info=seg)
            )

        merged = merge_short_segments(
            vocal,
            sample_rate,
            metadata,
            output_dir / "long_cut_wavs",
            max_duration_ms=max_merge_duration,
        )

        final_metadata = []

        for item in merged:
            self.f0_extractor.process(item.wav_fn, f0_path=item.wav_fn.replace(".wav", "_f0.npy"))
            final_metadata.append(convert_metadata(item))

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(final_metadata, f, ensure_ascii=False, indent=2)

        shutil.copy(output_dir / "metadata.json", audio_path.replace(".wav", ".json").replace(".mp3", ".json").replace(".flac", ".json"))


def main(args):
    pipeline = PreprocessPipeline(
        device=args.device,
        language=args.language,
        save_dir=args.save_dir,
        vocal_sep=args.vocal_sep,
        max_merge_duration=args.max_merge_duration,
    )
    pipeline.run(
        audio_path=args.audio_path,
        language=args.language
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the output files")
    parser.add_argument("--language", type=str, default="Mandarin", help="Language of the audio")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the models on")
    parser.add_argument("--vocal_sep", type=bool, default=True, help="Whether to perform vocal separation")
    parser.add_argument("--max_merge_duration", type=int, default=60000, help="Maximum merged segment duration in milliseconds")    
    args = parser.parse_args()

    main(args)
