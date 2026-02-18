# https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
import os
import re
import time
import logging
from typing import Any, Dict, List, Tuple

# Suppress verbose NeMo logging
logging.getLogger('nemo').setLevel(logging.ERROR)
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import librosa
import numpy as np
from funasr import AutoModel


def _build_words_with_gaps(raw_words, raw_timestamps, wav_fn: str):
    words, word_durs = [], []
    prev = 0.0
    for w, t in zip(raw_words, raw_timestamps):
        s, e = float(t[0]), float(t[1])
        if s > prev:
            words.append("<SP>")
            word_durs.append(s - prev)
        words.append(w)
        word_durs.append(e - s)
        prev = e

    wav_len = librosa.get_duration(filename=wav_fn)
    if wav_len > prev:
        if len(words) == 0:
            words.append("<SP>")
            word_durs.append(wav_len)
            return words, word_durs
        if words[-1] != "<SP>":
            words.append("<SP>")
            word_durs.append(wav_len - prev)
        else:
            word_durs[-1] += wav_len - prev

    return words, word_durs

def _word_dur_post_process(words, word_durs, f0):
    """Post-process word durations using f0 to better place silences.
    """
    # f0 time grid parameters
    sr = 24000  # f0 sample rate
    hop_length = 480  # f0 hop length

    # Convert word durations (seconds) to frame boundaries on the f0 grid.
    boundaries = np.cumsum([
        0,
        *[
            int(dur * sr / hop_length)
            for dur in word_durs
        ],
    ]).tolist()

    sil_tolerance = 5   # tolerance frames for silence detection
    ext_tolerance = 5   # tolerance frames for vocal extension

    new_words: list[str] = []
    new_word_durs: list[float] = []
    if words:
        new_words.append(words[0])
        new_word_durs.append(word_durs[0])

    for i in range(1, len(words)):
        word = words[i]
        if word == "<SP>":
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]

            num_frames = end_frame - start_frame
            frame_idx = start_frame

            # Find first region with at least 5 consecutive "unvoiced" frames.
            unvoiced_count = 0
            while frame_idx < end_frame:
                if f0[frame_idx] <= 1:  # unvoiced
                    unvoiced_count += 1
                    if unvoiced_count >= sil_tolerance:
                        frame_idx -= sil_tolerance - 1  # back to the last voiced frame
                        break
                else:
                    unvoiced_count = 0
                frame_idx += 1

            voice_frames = frame_idx - start_frame

            if voice_frames >= int(num_frames * 0.9):  # over 90% voiced
                # Treat the whole "<SP>" as silence and merge into previous word.
                new_word_durs[-1] += word_durs[i]
            elif voice_frames >= ext_tolerance:  # over 5 frames voiced
                # Split the "<SP>" into two parts: leading silence and tail kept as "<SP>".
                dur = voice_frames * hop_length / sr
                new_word_durs[-1] += dur
                new_words.append("<SP>")
                new_word_durs.append(word_durs[i] - dur)
            else:
                # Too short to adjust, keep as-is.
                new_words.append(word)
                new_word_durs.append(word_durs[i])
        else:
            new_words.append(word)
            new_word_durs.append(word_durs[i])

    return new_words, new_word_durs


class _ASRZhModel:
    """Mandarin/Cantonese ASR wrapper."""

    def __init__(self, model_path: str, device: str):
        self.model = AutoModel(
            model=model_path,
            disable_update=True,
            device=device,
        )

    def process(self, wav_fn):
        out = self.model.generate(wav_fn, output_timestamp=True)[0]
        raw_words = out["text"].replace("@", "").split(" ")
        raw_timestamps = [[t[0] / 1000, t[1] / 1000] for t in out["timestamp"]]
        words, word_durs = _build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        if os.path.exists(wav_fn.replace(".wav", "_f0.npy")):
            words, word_durs = _word_dur_post_process(
                words, word_durs, np.load(wav_fn.replace(".wav", "_f0.npy"))
            )

        return words, word_durs


class _ASREnModel:
    """English ASR wrapper for NeMo Parakeet-TDT.
    
    NOTE: Uses CPU to avoid Windows CUDA graph compatibility issues.
    """

    def __init__(self, model_path: str, device: str):
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "NeMo (nemo_toolkit) is required for ASR English but is not available in this Python env. "
                "Install it in the active environment, then retry."
            ) from e

        # Force CPU to avoid CUDA graph issues on Windows
        # This is a workaround for the CUDA graph compatibility issue
        cpu_device = "cpu"
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=model_path,
            map_location=cpu_device,
        )
        self.model.eval()
        self.device = cpu_device  # Store the actual device used

    @staticmethod
    def _clean_word(word: str) -> str:
        return re.sub(r"[\?\.,:]", "", word).strip()

    @staticmethod
    def _extract_word_segments(output: Any) -> List[Dict[str, Any]]:
        ts = getattr(output, "timestamp", None)
        if not ts or not isinstance(ts, dict):
            return []
        word_ts = ts.get("word")
        return word_ts if isinstance(word_ts, list) else []

    def process(self, wav_fn: str) -> Tuple[List[str], List[float]]:
        # Process on CPU to avoid Windows CUDA graph issues
        outputs = self.model.transcribe(
            [wav_fn],
            timestamps=True,
            batch_size=1,
            num_workers=0,
        )
        output = outputs[0] if outputs else None

        raw_words: List[str] = []
        raw_timestamps: List[List[float]] = []
        if output is not None:
            for w in self._extract_word_segments(output):
                s, e = float(w.get("start", 0.0)), float(w.get("end", 0.0))
                word = self._clean_word(str(w.get("word", "")))
                if word:
                    raw_words.append(word)
                    raw_timestamps.append([s, e])

        words, durs = _build_words_with_gaps(raw_words, raw_timestamps, wav_fn)

        if os.path.exists(wav_fn.replace(".wav", "_f0.npy")):
            words, durs = _word_dur_post_process(
                words, durs, np.load(wav_fn.replace(".wav", "_f0.npy"))
            )

        return words, durs


class LyricTranscriber:
    """Transcribe lyrics from singing voice segment
    """

    def __init__(
        self,
        zh_model_path: str,
        en_model_path: str,
        device: str = "cuda",
        *,
        verbose: bool = True,
    ):
        """Initialize lyric transcriber.

        Args:
            zh_model_path (str): Path to the Chinese model file.
            en_model_path (str): Path to the English model file.
            device (str): Device to use for tensor operations.
            verbose (bool): Whether to print verbose logs.
        """
        self.verbose = verbose
        self.device = device
        self.zh_model_path = zh_model_path
        self.en_model_path = en_model_path

        if self.verbose:
            print(
                "[lyric transcription] init: start:",
                f"device={device}",
                f"model_path={zh_model_path}",
            )

        # Always initialize Chinese ASR.
        self.zh_model = _ASRZhModel(device=device, model_path=zh_model_path)

        # English ASR will be lazily initialized on first English request to avoid long waiting cost when importing NeMo
        self.en_model = None

        if self.verbose:
            print("[lyric transcription] init: success")

    def process(self, wav_fn, language: str | None = "Mandarin", *, verbose: bool | None = None):
        """ Lyric transcriber process

        Args:
            wav_fn (str): Path to the audio file.
            language (str | None): Language of the audio. Defaults to "Mandarin". Supports "Mandarin", "Cantonese" and "English".
            verbose (bool | None): Whether to print verbose logs. Defaults to None.
        """
        v = self.verbose if verbose is None else verbose
        if language not in {"Mandarin", "Cantonese", "English"}:
            raise ValueError(f"Unsupported language: {language}, should be one of ['Mandarin', 'Cantonese', 'English']")
        if v:
            print(f"[lyric transcription] process: start: wav_fn={wav_fn} language={language}")
            t0 = time.time()

        lang = (language or "auto").lower()
        if lang in {"english"}:
            if self.en_model is None:
                # Lazy-load NeMo model only when English is actually used.
                if v:
                    print("[lyric transcription] init English ASR, please make sure NeMo is installed")
                self.en_model = _ASREnModel(model_path=self.en_model_path, device=self.device)
            out = self.en_model.process(wav_fn)
        else:
            out = self.zh_model.process(wav_fn)

        if v:
            words, durs = out
            n_words = len(words) if isinstance(words, list) else 0
            dur_sum = float(sum(durs)) if isinstance(durs, list) else 0.0
            dt = time.time() - t0
            print(
                "[lyric transcription] process: done:",
                f"n_words={n_words}",
                f"dur_sum={dur_sum:.3f}s",
                f"time={dt:.3f}s",
            )

        return out


if __name__ == "__main__":
    m = LyricTranscriber(
        zh_model_path="models/SoulX-Singer/preprocessors/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        en_model_path="models/SoulX-Singer/preprocessors/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        device="cuda"
    )
    print(m.process("example/test/asr_zh.wav", language="Mandarin"))
    print(m.process("example/test/asr_en.wav", language="English"))