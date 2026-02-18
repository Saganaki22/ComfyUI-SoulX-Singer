# https://github.com/ZFTurbo/Music-Source-Separation-Training
# https://huggingface.co/becruily/mel-band-roformer-karaoke/blob/main/mel_band_roformer_karaoke_becruily.ckpt
# https://huggingface.co/anvuew/dereverb_mel_band_roformer/blob/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import librosa
import sys
import os
import time
import torch
import numpy as np

from .utils.audio_utils import normalize_audio, denormalize_audio
from .utils.settings import get_model_from_config, parse_args_inference
from .utils.model_utils import demix
from .utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint


def process(mix, model, args, config, device):

    instruments = prefer_target_instrument(config)[:]

    # If mono audio we must adjust it depending on model
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                # print(f'Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

    if 'normalize' in config.inference:
        if config.inference['normalize'] is True:
            mix, norm_params = normalize_audio(mix)

    waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=not args.disable_detailed_pbar)

    instr = 'vocals' if 'vocals' in instruments else instruments[0]
    estimates = waveforms_orig[instr]
    if 'normalize' in config.inference:
        if config.inference['normalize'] is True:
            estimates = denormalize_audio(estimates, norm_params)

    return estimates


def build_model(args):
    model, config = get_model_from_config(args.model_type, args.config_path)
    
    load_start_checkpoint(args, model, None, type_='inference')

    return model, config


def build_models(dict_args):
    args = parse_args_inference(dict_args)

    ########## load model ##########
    torch.backends.cudnn.benchmark = True

    args.config_path = args.sep_config_path
    args.start_check_point = args.sep_start_check_point

    sep_model, sep_config = build_model(args)

    args.config_path = args.der_config_path
    args.start_check_point = args.der_start_check_point
    
    dereverb_model, dereverb_config = build_model(args)

    sep_model = sep_model
    dereverb_model = dereverb_model

    return sep_model, sep_config, dereverb_model, dereverb_config, args

def main(args, sep_model=None, sep_config=None, dereverb_model=None, dereverb_config=None, device=None):

    ######## process data ##########
    sample_rate = getattr(sep_config.audio, 'sample_rate', 44100)
    path = args.input_path

    mix, _ = librosa.load(path, sr=sample_rate, mono=False)
    vocals = process(mix, sep_model, args, sep_config, device)
    dereverbed_vocals = process(vocals.mean(0), dereverb_model, args, dereverb_config, device)
    accompaniment = mix - dereverbed_vocals

    return mix, vocals, dereverbed_vocals, accompaniment, sample_rate

@dataclass
class VocalSeparationOutputs:
    """Vocal extraction output container."""

    mix: np.ndarray
    vocals: np.ndarray
    vocals_dereverbed: np.ndarray
    accompaniment: np.ndarray
    sample_rate: int


class VocalSeparator:
    """Vocal separation and dereverb wrapper.

    Wraps the karaoke separation and dereverb models from the
    ZFTurbo Music Source Separation project and exposes a simple
    :py:meth:`process` API that returns mix/vocals/dereverbed/accompaniment.
    """
    def __init__(
        self,
        sep_model_path: str,
        sep_config_path: str,
        der_model_path: str,
        der_config_path: str,
        *,
        model_type: str = "mel_band_roformer",
        disable_detailed_pbar: bool = True,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """Initialize the vocal separator.

        Args:
            device: Torch device string, e.g. ``"cuda:0"``.
            model_type: Separation model type key.
            sep_config_path: Config path for separation model.
            sep_start_check_point: Checkpoint path for separation model.
            der_config_path: Config path for dereverb model.
            der_start_check_point: Checkpoint path for dereverb model.
            disable_detailed_pbar: Disable detailed progress bars in underlying utils.
            verbose: Whether to print verbose logs.
        """

        # Match original script args schema
        args_dict: Dict[str, Any] = {
            "model_type": model_type,
            "disable_detailed_pbar": disable_detailed_pbar,
            "sep_config_path": sep_config_path,
            "sep_start_check_point": sep_model_path,
            "der_config_path": der_config_path,
            "der_start_check_point": der_model_path,
        }

        if verbose:
            print("[vocal extraction] init: start")

        sep_model, sep_config, dereverb_model, dereverb_config, args = build_models(args_dict)

        sep_model = sep_model.to(device)
        dereverb_model = dereverb_model.to(device)

        self.sep_model = sep_model
        self.sep_config = sep_config
        self.dereverb_model = dereverb_model
        self.dereverb_config = dereverb_config
        self.device = device
        self.args = args
        self.verbose = verbose

        if verbose:
            print(
                "[vocal extraction] init success: sep=loaded, dereverb=loaded, device=",
                device,
            )

    def process(self, input_path: str, *, verbose: Optional[bool] = None) -> VocalSeparationOutputs:
        """Separate a single audio file into sources.

        Args:
            input_path: Path to the mixture wav.
            verbose: Override instance-level verbose flag for this call.

        Returns:
            :class:`VocalSeparationOutputs` containing mix, vocals,
            dereverbed vocals, accompaniment and sample rate.
        """
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[vocal extraction] process_file: start: {input_path}")
            t0 = time.time()

        self.args.input_path = input_path

        mix, vocals, dereverbed, accompaniment, sample_rate = main(
            self.args,
            self.sep_model,
            self.sep_config,
            self.dereverb_model,
            self.dereverb_config,
            torch.device(self.device) if not isinstance(self.device, torch.device) else self.device,
        )

        if verbose:
            dt = time.time() - t0
            print(
                "[vocal extraction] process_file: done:",
                f"sr={sample_rate}",
                f"mix={getattr(mix, 'shape', None)}",
                f"vocals={getattr(vocals, 'shape', None)}",
                f"dereverbed={getattr(dereverbed, 'shape', None)}",
                f"acc={getattr(accompaniment, 'shape', None)}",
                f"time={dt:.3f}s",
            )

        return VocalSeparationOutputs(
            mix=mix,
            vocals=vocals,
            vocals_dereverbed=dereverbed,
            accompaniment=accompaniment,
            sample_rate=sample_rate,
        )


if __name__ == "__main__":
    
    m = VocalSeparator(
        sep_model_path="models/SoulX-Singer/preprocessors/mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt",
        sep_config_path="models/SoulX-Singer/preprocessors/mel-band-roformer-karaoke/config_karaoke_becruily.yaml",
        der_model_path="models/SoulX-Singer/preprocessors/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        der_config_path="models/SoulX-Singer/preprocessors/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml",
        device="cuda"
    )

    out = m.process("example/test/separation_test.mp3")
    print(out.vocals_dereverbed.shape)
