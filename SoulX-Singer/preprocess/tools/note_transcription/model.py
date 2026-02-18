# https://github.com/RickyL-2000/ROSVOT
import math
import sys
import traceback
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt

from .utils.os_utils import safe_path
from .utils.commons.hparams import set_hparams
from .utils.commons.ckpt_utils import load_ckpt
from .utils.commons.dataset_utils import pad_or_cut_xd
from .utils.audio.mel import MelNet
from .utils.audio.pitch_utils import (
    norm_interp_f0,
    denorm_f0,
    f0_to_coarse,
    boundary2Interval,
    save_midi,
    midi_to_hz,
)
from .utils.rosvot_utils import (
    get_mel_len,
    align_word,
    regulate_real_note_itv,
    regulate_ill_slur,
    bd_to_durs,
)
from .modules.pe.rmvpe import RMVPE
from .modules.rosvot.rosvot import MidiExtractor, WordbdExtractor


@torch.no_grad()
def infer_sample(
    item: Dict[str, Any],
    hparams: Dict[str, Any],
    models: Dict[str, Any],
    device: torch.device,
    *,
    save_dir: Optional[str] = None,
    apply_rwbd: Optional[bool] = None,
    # outputs
    save_plot: bool = False,
    no_save_midi: bool = True,
    no_save_npy: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    if "item_name" not in item or "wav_fn" not in item:
        raise ValueError('item must contain keys: "item_name" and "wav_fn"')
    
    item_name = item["item_name"]
    wav_src = item["wav_fn"]

    # Decide RWBD usage
    if apply_rwbd is None:
        apply_rwbd_ = ("word_durs" not in item)
    else:
        apply_rwbd_ = bool(apply_rwbd)

    # Models
    model = models["model"]
    mel_net = models["mel_net"]
    pe = models.get("pe")
    wbd_predictor = models.get("wbd_predictor")

    if wbd_predictor is None and apply_rwbd_:
         raise ValueError("apply_rwbd is True but wbd_predictor model is not provided in models")

    # ---- Prepare Data  ----
    if isinstance(wav_src, str):
        wav, _ = librosa.core.load(wav_src, sr=hparams["audio_sample_rate"])
    else:
        wav = wav_src
        if not isinstance(wav, np.ndarray):
            wav = np.asarray(wav)
    wav = wav.astype(np.float32)

    # Calculate timestamps and alignment lengths
    wav_len_samples = wav.shape[-1]
    mel_len = get_mel_len(wav_len_samples, hparams["hop_size"])
    
    # Word boundary preparation
    mel2word = None
    word_durs_filtered = None
    
    if not apply_rwbd_:
        if "word_durs" not in item:
             raise ValueError('apply_rwbd=False but item has no "word_durs"')
        
        wd_raw = list(item["word_durs"]) 
        min_word_dur = hparams.get("min_word_dur", 20) / 1000
        word_durs_filtered = []
        
        for i, wd in enumerate(wd_raw):
            if wd < min_word_dur:
                if i == 0 and len(wd_raw) > 1:
                    wd_raw[i + 1] += wd
                elif len(word_durs_filtered) > 0:
                    word_durs_filtered[-1] += wd
            else:
                word_durs_filtered.append(wd)

        mel2word, _ = align_word(word_durs_filtered, mel_len, hparams["hop_size"], hparams["audio_sample_rate"])
        mel2word = np.asarray(mel2word)
        if mel2word.size > 0 and mel2word[0] == 0:
             mel2word = mel2word + 1
        
        mel2word_len = int(np.sum(mel2word > 0))
        real_len = min(mel_len, mel2word_len)
    else:
        real_len = min(mel_len, hparams["max_frames"])

    T = math.ceil(min(real_len, hparams["max_frames"]) / hparams["frames_multiple"]) * hparams["frames_multiple"]

    # ---- Input Tensors & Padding ----
    target_samples = T * hparams["hop_size"]
    wav_t = torch.from_numpy(wav).float().to(device).unsqueeze(0) # [1, L]
    if wav_t.shape[-1] < target_samples:
        wav_t = pad_or_cut_xd(wav_t, target_samples, 1)
    
    # ---- Pitch Extraction ----
    if pe is not None:
        f0s, uvs = pe.get_pitch_batch(
            wav_t,
            sample_rate=hparams["audio_sample_rate"],
            hop_size=hparams["hop_size"],
            lengths=[real_len], 
            fmax=hparams["f0_max"],
            fmin=hparams["f0_min"],
        )
        f0_1d, uv_1d = norm_interp_f0(f0s[0][:T])
        f0_t = pad_or_cut_xd(torch.FloatTensor(f0_1d).to(device), T, 0).unsqueeze(0)
        uv_t = pad_or_cut_xd(torch.FloatTensor(uv_1d).to(device), T, 0).long().unsqueeze(0)
        pitch_coarse = f0_to_coarse(denorm_f0(f0_t, uv_t)).to(device)
        f0_np = denorm_f0(f0_t, uv_t)[0].detach().cpu().numpy()[:real_len]
    else:
        f0_t = uv_t = pitch_coarse = None
        f0_np = None

    # ---- Mel Extraction ----
    mel = mel_net(wav_t) # [1, T_padded, C]
    mel = pad_or_cut_xd(mel, T, 1)
    
    # Construct non-padding mask
    mel_nonpadding_mask = torch.zeros(1, T, device=device)
    mel_nonpadding_mask[:, :real_len] = 1.0
    
    # Apply mask to mel (zero out padding)
    mel = (mel.transpose(1, 2) * mel_nonpadding_mask.unsqueeze(1)).transpose(1, 2)
    # Re-calculate non_padding bool mask
    mel_nonpadding = mel.abs().sum(-1) > 0

    # ---- Word Boundary ----
    word_durs_used = None
    if apply_rwbd_:
        mel_input = mel[:, :, : hparams.get("wbd_use_mel_bins", 80)]
        wbd_outputs = wbd_predictor(
            mel=mel_input,
            pitch=pitch_coarse,
            uv=uv_t,
            non_padding=mel_nonpadding,
            train=False,
        )
        word_bd = wbd_outputs["word_bd_pred"] # [1, T]
    else:
        # Construct word_bd from provided durs
        mel2word_t = pad_or_cut_xd(torch.LongTensor(mel2word).to(device), T, 0)
        word_bd = torch.zeros_like(mel2word_t)
        # Vectorized check
        word_bd[1:] = (mel2word_t[1:] != mel2word_t[:-1]).long()
        word_bd[real_len:] = 0
        word_bd = word_bd.unsqueeze(0) # [1, T]
        
        word_durs_used = np.array(word_durs_filtered)

    # ---- Main Inference ----
    mel_input = mel[:, :, : hparams.get("use_mel_bins", 80)]
    outputs = model(
        mel=mel_input,
        word_bd=word_bd,
        pitch=pitch_coarse,
        uv=uv_t,
        non_padding=mel_nonpadding,
        train=False,
    )

    note_lengths = outputs["note_lengths"].detach().cpu().numpy()
    note_bd_pred = outputs["note_bd_pred"][0].detach().cpu().numpy()[:real_len]
    note_pred = outputs["note_pred"][0].detach().cpu().numpy()[: note_lengths[0]]
    note_bd_logits = torch.sigmoid(outputs["note_bd_logits"])[0].detach().cpu().numpy()[:real_len]

    if note_pred.shape == (0,):
        if verbose:
            print(f"skip {item_name}: no notes detected")
        return {
            "item_name": item_name,
            "pitches": [],
            "note_durs": [],
            "note2words": None,
        }

    # ---- Post-Processing & Regulation ----
    note_itv_pred = boundary2Interval(note_bd_pred)
    note2words = None

    if apply_rwbd_:
        word_bd_np = outputs['word_bd_pred'][0].detach().cpu().numpy()[:real_len]
        word_durs_derived = np.array(bd_to_durs(word_bd_np)) * hparams['hop_size'] / hparams['audio_sample_rate']
        word_durs_for_reg = word_durs_derived
        word_bd_for_reg = word_bd_np
    else:
        word_bd_for_reg = word_bd[0].detach().cpu().numpy()[:real_len]
        word_durs_for_reg = word_durs_used

    should_regulate = hparams.get("infer_regulate_real_note_itv", True) and (not apply_rwbd_)
    
    if should_regulate and (word_durs_for_reg is not None):
        try:
            note_itv_pred_secs, note2words = regulate_real_note_itv(
                note_itv_pred,
                note_bd_pred,
                word_bd_for_reg,
                word_durs_for_reg,
                hparams["hop_size"],
                hparams["audio_sample_rate"],
            )
            note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(note_pred, note_itv_pred_secs, note2words)
        except Exception as err:
            if verbose:
                _, exc_value, exc_tb = sys.exc_info()
                tb = traceback.extract_tb(exc_tb)[-1]
                print(f"postprocess failed: {err}: {exc_value} in {tb[0]}:{tb[1]} '{tb[2]}' in {tb[3]}")
            # Fallback
            note_itv_pred_secs = note_itv_pred * hparams["hop_size"] / hparams["audio_sample_rate"]
            note2words = None
    else:
        note_itv_pred_secs = note_itv_pred * hparams["hop_size"] / hparams["audio_sample_rate"]

    # ---- Output ----
    note_durs = [float((itv[1] - itv[0])) for itv in note_itv_pred_secs]
    
    out = {
        "item_name": item_name,
        "pitches": note_pred.tolist(),
        "note_durs": note_durs,
        "note2words": note2words.tolist() if note2words is not None else None,
    }

    # ---- Saving ----
    if save_dir is not None:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        fn = str(item_name)
        
        if not no_save_midi:
            save_midi(note_pred, note_itv_pred_secs, safe_path(save_dir_path / "midi" / f"{fn}.mid"))
        
        if not no_save_npy:
            np.save(safe_path(save_dir_path / "npy" / f"[note]{fn}.npy"), out, allow_pickle=True)

        if save_plot:
            fig = plt.figure()
            if f0_np is not None:
                plt.plot(f0_np, color="red", label="f0")

            midi_pred = np.zeros(note_bd_pred.shape[0], dtype=np.float32)
            itvs = np.round(note_itv_pred_secs * hparams["audio_sample_rate"] / hparams["hop_size"]).astype(int)
            for i, itv in enumerate(itvs):
                midi_pred[itv[0] : itv[1]] = note_pred[i]
            plt.plot(midi_to_hz(midi_pred), color="blue", label="pred midi")
            plt.plot(note_bd_logits * 100, color="green", label="note bd logits x100")
            plt.legend()
            plt.tight_layout()
            plt.savefig(safe_path(save_dir_path / "plot" / f"[MIDI]{fn}.png"), format="png")
            plt.close(fig)

    return out


def load_rosvot_models(ckpt, config="", wbd_ckpt="", wbd_config="", device="cuda:0", verbose=False, thr=0.85):
    """
    Load models once to reuse across multiple items.
    """
    dev = torch.device(device)
    
    # 1. Hparams
    config_path = Path(ckpt).with_name("config.yaml") if config == "" else config
    pe_ckpt = Path(ckpt).parent.parent / "rmvpe/model.pt"
    hparams = set_hparams(
        config=config_path,
        print_hparams=verbose,
        hparams_str=f"note_bd_threshold={thr}",
    )
    
    # 2. Main Model
    model = MidiExtractor(hparams)
    load_ckpt(model, ckpt, verbose=verbose)
    model.eval().to(dev)

    # 3. MelNet
    mel_net = MelNet(hparams)
    mel_net.to(dev)

    # 4. Pitch Extractor
    pe = None
    if hparams.get("use_pitch_embed", False):
        pe = RMVPE(pe_ckpt, device=dev)
        
    # 5. Word Boundary Predictor (optional but we load if ckpt provided or needed)
    wbd_predictor = None
    if wbd_ckpt:
        wbd_config_path = Path(wbd_ckpt).with_name("config.yaml") if wbd_config == "" else wbd_config
        wbd_hparams = set_hparams(
            config=wbd_config_path,
            print_hparams=False,
            hparams_str="",
        )
        hparams.update({
            "wbd_use_mel_bins": wbd_hparams["use_mel_bins"],
            "min_word_dur": wbd_hparams["min_word_dur"],
        })
        wbd_predictor = WordbdExtractor(wbd_hparams)
        load_ckpt(wbd_predictor, wbd_ckpt, verbose=verbose)
        wbd_predictor.eval().to(dev)

    models = {
        "model": model,
        "mel_net": mel_net,
        "pe": pe,
        "wbd_predictor": wbd_predictor
    }
    return hparams, models


class NoteTranscriber:
    """Note transcription wrapper based on ROSVOT.

    Loads ROSVOT and optional RWBD models once in ``__init__`` and
    exposes a :py:meth:`process` API that turns an item dict into
    aligned note metadata for downstream SVS.
    """

    def __init__(
        self,
        rosvot_model_path: str,
        rwbd_model_path: str,
        *,
        rosvot_config_path: str = "",
        rwbd_config_path: str = "",
        device: str = "cuda:0",
        thr: float = 0.85,
        verbose: bool = True,
    ):
        """Initialize the note transcriber.

        Args:
            ckpt: Path to the main ROSVOT checkpoint.
            config: Optional config YAML path for ROSVOT.
            wbd_ckpt: Optional word-boundary checkpoint path.
            wbd_config: Optional config YAML path for RWBD.
            device: Torch device string, e.g. ``"cuda:0"`` / ``"cpu"``.
            thr: Note boundary threshold.
            verbose: Whether to print verbose logs.
        """
        self.verbose = verbose
        self.device = torch.device(device)
        self.hparams, self.models = load_rosvot_models(
            ckpt=rosvot_model_path,
            config=rosvot_config_path,
            wbd_ckpt=rwbd_model_path,
            wbd_config=rwbd_config_path,
            device=device,
            verbose=verbose,
            thr=thr,
        )

        if self.verbose:
            print(
                "[note transcription] init success:",
                f"device={self.device}",
                f"rosvot_model_path={rosvot_model_path}",
                f"rwbd_model_path={rwbd_model_path if rwbd_model_path else 'None'}",
                f"thr={thr}",
            )

    def process(
        self,
        item: Dict[str, Any],
        *,
        segment_info: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        apply_rwbd: Optional[bool] = None,
        save_plot: bool = False,
        no_save_midi: bool = True,
        no_save_npy: bool = True,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run ROSVOT on a single item and post-process outputs.

        Args:
            item: Input metadata dict with at least ``item_name`` and ``wav_fn``.
            segment_info: Optional segment metadata for sliced audio.
            save_dir: Optional directory for debug artifacts (plots, midis).
            apply_rwbd: Whether to run RWBD-based word boundary refinement.
            save_plot: Whether to save diagnostic plots.
            no_save_midi: If True, skip saving midi.
            no_save_npy: If True, skip saving numpy intermediates.
            verbose: Override instance-level verbose flag for this call.

        Returns:
            Dict with aligned note information for downstream SVS.
        """
        v = self.verbose if verbose is None else verbose
        if v:
            item_name = item.get("item_name", "")
            wav_fn = item.get("wav_fn", "")
            print(f"[note transcription] process: start: item_name={item_name} wav_fn={wav_fn}")
            t0 = time.time()

        rosvot_out = infer_sample(
            item,
            self.hparams,
            self.models,
            device=self.device,
            save_dir=save_dir,
            apply_rwbd=apply_rwbd,
            save_plot=save_plot,
            no_save_midi=no_save_midi,
            no_save_npy=no_save_npy,
            verbose=v,
        )

        out = self.post_process(
            metadata=item,
            segment_info=segment_info,
            rosvot_out=rosvot_out,
        )

        if v:
            dt = time.time() - t0
            print(
                "[note transcription] process: done:",
                f"item_name={out.get('item_name','')}",
                f"n_notes={len(out.get('note_pitch', []) or [])}",
                f"time={dt:.3f}s",
            )

        return out

    @staticmethod
    def _normalize_note2words(note2words: list[int]) -> list[int]:
        if not note2words:
            return []
        normalized = [note2words[0]]
        for idx in range(1, len(note2words)):
            if note2words[idx] < normalized[-1]:
                normalized.append(normalized[-1])
            else:
                normalized.append(note2words[idx])
        return normalized

    @staticmethod
    def _build_ep_types(note2words: list[int], align_words: list[str]) -> list[int]:
        ep_types: list[int] = []
        prev = -1
        for i, w in zip(note2words, align_words):
            if w == "<SP>":
                ep_types.append(1)
            else:
                ep_types.append(2 if i != prev else 3)
            prev = i
        return ep_types

    def post_process(
        self,
        *,
        metadata: Dict[str, Any],
        segment_info: Dict[str, Any],
        rosvot_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build aligned note metadata using ROSVOT outputs."""
        note2words_raw = rosvot_out.get("note2words") or []
        note2words = self._normalize_note2words(note2words_raw)
        align_words = [
            metadata["words"][idx - 1]
            for idx in note2words_raw
            if 0 < idx <= len(metadata["words"])
        ]
        ep_types = self._build_ep_types(note2words, align_words) if align_words else []

        return {
            "item_name": rosvot_out.get("item_name", "") if not segment_info else segment_info["item_name"],
            "wav_fn": metadata.get("wav_fn", "") if not segment_info else segment_info["wav_fn"],
            "origin_wav_fn": metadata.get("origin_wav_fn", "") if not segment_info else segment_info["origin_wav_fn"],
            "start_time_ms": "" if not segment_info else segment_info["start_time_ms"],
            "end_time_ms": "" if not segment_info else segment_info["end_time_ms"],
            "language": metadata.get("language", ""),
            "note_text": align_words,
            "note_dur": rosvot_out.get("note_durs", []),
            "note_type": ep_types,
            "note_pitch": rosvot_out.get("pitches", []),
        }

if __name__ == "__main__":

    items = json.load(open("example/test/rosvot_input.json", "r"))
    item = items[0]

    m = NoteTranscriber(
        rosvot_model_path="models/SoulX-Singer/preprocessors/rosvot/rosvot/model.pt", 
        rwbd_model_path="models/SoulX-Singer/preprocessors/rosvot/rwbd/model.pt", 
        device="cuda"
    )
    out = m.process(item)

    print(out)