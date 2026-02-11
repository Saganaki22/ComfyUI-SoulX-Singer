import os
import time
from dataclasses import dataclass
from typing import List, Optional

import librosa
import numpy as np
from soundfile import write


@dataclass(frozen=True)
class VocalDetectionConfig:
    hop_ms: int = 20
    smooth_ms: int = 200
    start_ms: int = 120
    end_ms: int = 200
    prepad_ms: int = 80
    postpad_ms: int = 120
    min_len_ms: int = 1000
    max_len_ms: int = 20000
    short_seg_merge_gap_ms: int = 8000
    small_gap_ms: int = 500
    lookback_ms: int = 200
    lookahead_ms: int = 100


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def _merge_short_segments(
    segments_ms: List[List[int]],
    *,
    min_len_ms: int,
    max_len_ms: int,
    short_seg_merge_gap_ms: int,
    small_gap_ms: int,
) -> List[List[int]]:
    if not segments_ms:
        return []

    merged: List[List[int]] = []
    cur_start, cur_end = segments_ms[0]

    for next_start, next_end in segments_ms[1:]:
        cur_len = cur_end - cur_start
        gap_ms = next_start - cur_end
        merged_len = next_end - cur_start

        should_merge = (
            (cur_len < min_len_ms and gap_ms < short_seg_merge_gap_ms)
            or (gap_ms < small_gap_ms and merged_len < max_len_ms)
        )

        if should_merge:
            cur_end = next_end
            continue

        if (cur_end - cur_start) >= min_len_ms:
            merged.append([cur_start, cur_end])

        cur_start, cur_end = next_start, next_end

    if (cur_end - cur_start) >= min_len_ms:
        merged.append([cur_start, cur_end])

    if not merged:
        return segments_ms

    return merged


def _voiced_to_segments(
    voiced: np.ndarray,
    *,
    hop_ms: int,
    smooth_ms: int,
    start_ms: int,
    end_ms: int,
    prepad_ms: int,
    postpad_ms: int,
    max_len_ms: int,
) -> List[List[int]]:
    smooth_frames = max(1, int(round(smooth_ms / hop_ms)))
    smooth_voiced = _moving_average(voiced.astype(np.float32), smooth_frames)
    active = smooth_voiced >= 0.5

    segments: List[List[int]] = []
    start_idx = None
    start_frames = max(1, int(round(start_ms / hop_ms)))
    end_frames = max(1, int(round(end_ms / hop_ms)))
    prepad_frames = max(0, int(round(prepad_ms / hop_ms)))
    postpad_frames = max(0, int(round(postpad_ms / hop_ms)))
    active_count = 0
    inactive_count = 0

    for i, flag in enumerate(active):
        if flag:
            active_count += 1
            inactive_count = 0
        else:
            inactive_count += 1
            active_count = 0

        if start_idx is None:
            if active_count >= start_frames:
                start_idx = max(0, i - start_frames + 1 - prepad_frames)
        else:
            if inactive_count >= end_frames:
                end_idx = min(len(active) - 1, i - end_frames + 1 + postpad_frames)
                start_ms_val = start_idx * hop_ms
                end_ms_val = end_idx * hop_ms + hop_ms
                if end_ms_val > start_ms_val:
                    segments.append([int(start_ms_val), int(end_ms_val)])
                start_idx = None

    if start_idx is not None:
        start_ms_val = start_idx * hop_ms
        end_idx = min(len(active) - 1, len(active) - 1 + postpad_frames)
        end_ms_val = end_idx * hop_ms + hop_ms
        if end_ms_val > start_ms_val:
            segments.append([int(start_ms_val), int(end_ms_val)])

    def _split_segment(seg: List[int]) -> List[List[int]]:
        start_ms_val, end_ms_val = seg
        start_frame = int(start_ms_val // hop_ms)
        end_frame = int((end_ms_val - 1) // hop_ms)
        end_frame = max(start_frame, min(end_frame, len(active) - 1))

        best_start = None
        best_len = 0
        cur_start = None
        cur_len = 0
        for idx in range(start_frame, end_frame + 1):
            if not active[idx]:
                if cur_start is None:
                    cur_start = idx
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if cur_start is not None and cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_start = None
                cur_len = 0
        if cur_start is not None and cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_start is None:
            split_frame = (start_frame + end_frame) // 2
        else:
            split_frame = best_start + best_len // 2

        split_ms = split_frame * hop_ms
        if split_ms <= start_ms_val:
            split_ms = start_ms_val + hop_ms
        if split_ms >= end_ms_val:
            split_ms = end_ms_val - hop_ms

        if split_ms <= start_ms_val or split_ms >= end_ms_val:
            return [seg]

        return [[start_ms_val, int(split_ms)], [int(split_ms), end_ms_val]]

    queue = segments[:]
    segments = []
    while queue:
        seg = queue.pop(0)
        if (seg[1] - seg[0]) <= max_len_ms:
            segments.append(seg)
            continue
        parts = _split_segment(seg)
        if len(parts) == 1:
            segments.append(seg)
        else:
            queue = parts + queue

    return segments


class VocalDetector:
    """Detect vocal segments based on f0 voiced decisions.

    This component consumes a precomputed ``*_f0.npy`` track and
    produces vocal segments (and cuts wav files) for downstream
    transcription or singing voice tasks.
    """
    def __init__(
        self,
        cut_wavs_output_dir: str = "cut_wavs",
        config: VocalDetectionConfig | None = None,
        *,
        verbose: bool = True,
    ):
        """Initialize the vocal detector.

        Args:
            cut_wavs_output_dir: Directory to save cut wav segments.
            config: Detection configuration; uses :class:`VocalDetectionConfig` by default.
            verbose: Whether to print verbose logs.
        """
        self.cut_wavs_output_dir = cut_wavs_output_dir
        self.config = config or VocalDetectionConfig()
        self.verbose = verbose

        if self.verbose:
            print(
                "[vocal detection] init success:",
                f"cut_wavs_output_dir={self.cut_wavs_output_dir}",
                f"hop_ms={self.config.hop_ms}",
            )

    def process(self, audio_path: str, f0: np.ndarray, *, verbose: Optional[bool] = None) -> List[dict]:
        """Run vocal detection on a single wav.

        Args:
            audio_path: Path to the input wav file.
            f0: The f0 contour to use for vocal detection.
            verbose: Override instance-level verbose flag for this call.

        Returns:
            A list of segment metadata dicts with fields like
            ``item_name``, ``wav_fn``, ``start_time_ms``, ``end_time_ms``.
        """
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[vocal detection] process: start: {audio_path}")
            t0 = time.time()

        os.makedirs(self.cut_wavs_output_dir, exist_ok=True)

        base_name = os.path.basename(audio_path)
        base_name_no_ext = os.path.splitext(base_name)[0]

        voiced = f0 > 0

        segments_ms = _voiced_to_segments(
            voiced,
            hop_ms=self.config.hop_ms,
            smooth_ms=self.config.smooth_ms,
            start_ms=self.config.start_ms,
            end_ms=self.config.end_ms,
            prepad_ms=self.config.prepad_ms,
            postpad_ms=self.config.postpad_ms,
            max_len_ms=self.config.max_len_ms,
        )

        if verbose:
            print(f"[vocal detection] segments(before_merge)={len(segments_ms)}")

        segments_ms = _merge_short_segments(
            segments_ms,
            min_len_ms=self.config.min_len_ms,
            max_len_ms=self.config.max_len_ms,
            short_seg_merge_gap_ms=self.config.short_seg_merge_gap_ms,
            small_gap_ms=self.config.small_gap_ms,
        )

        if verbose:
            print(f"[vocal detection] segments(after_merge)={len(segments_ms)}")

        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Apply global lookback/lookahead in milliseconds
        lookback_ms = self.config.lookback_ms
        lookahead_ms = self.config.lookahead_ms

        adjusted_segments: List[List[int]] = []
        prev_end = 0
        for start_ms, end_ms in segments_ms:
            start_ms = max(0, start_ms - lookback_ms)
            end_ms = min(end_ms + lookahead_ms, int(y.shape[0] / sr * 1000))

            # Enforce non-overlap with previous segment, move backward the previous one.
            if start_ms < prev_end and len(adjusted_segments) > 0:
                adjusted_segments[-1][1] = start_ms

            adjusted_segments.append([start_ms, end_ms])
            prev_end = end_ms

        segment_infos = []
        for idx, (start_ms, end_ms) in enumerate(adjusted_segments):
            if end_ms - start_ms > self.config.max_len_ms:
                start_ms = end_ms - self.config.max_len_ms

            key = f"{base_name_no_ext}_{idx}"
            start_sample = librosa.time_to_samples(start_ms / 1000, sr=sr)
            end_sample = librosa.time_to_samples(end_ms / 1000, sr=sr)
            segment = y[start_sample:end_sample]

            write(f"{self.cut_wavs_output_dir}/{key}.wav", segment, sr)
            segment_infos.append(
                {
                    "item_name": key,
                    "wav_fn": f"{self.cut_wavs_output_dir}/{key}.wav",
                    "start_time_ms": int(start_sample * 1000 / sr),
                    "end_time_ms": int(end_sample * 1000 / sr),
                    "origin_wav_fn": audio_path,
                    "duration": int((end_sample - start_sample) * 1000 / sr),
                }
            )

        if verbose:
            dt = time.time() - t0
            print(
                "[vocal detection] process: done:",
                f"n_segments={len(segment_infos)}",
                f"time={dt:.3f}s",
            )

        return segment_infos


if __name__ == "__main__":
    m = VocalDetector(cut_wavs_output_dir="outputs/transcription/cut_wavs")
    segment_infos = m.process("./outputs/transcription/test.wav")
    print(segment_infos)
