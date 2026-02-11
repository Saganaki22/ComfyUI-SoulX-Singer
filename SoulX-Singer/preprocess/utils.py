import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import soundfile as sf
from dataclasses import dataclass

from preprocess.tools.g2p import g2p_transform


@dataclass
class SegmentMetadata:
    item_name: str
    wav_fn: str
    language: str
    start_time_ms: int
    end_time_ms: int
    note_text: List[str]
    note_dur: List[float]
    note_pitch: List[int]
    note_type: List[int]
    origin_wav_fn: Optional[str] = None


def _merge_group(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[SegmentMetadata],
    output_dir: Path,
    end_extension_ms: int = 0,
) -> SegmentMetadata:
    """
    Merge a group of consecutive segments into a single segment.

    This function:
    - Concatenates note-level information
    - Inserts <SP> for silence gaps
    - Merges consecutive <SP>
    - Cuts and writes merged audio
    - Determines dominant language

    Args:
        audio: Full vocal audio waveform (T,)
        sample_rate: Audio sample rate
        segments: Consecutive segments to be merged (SegmentMetadata or dict)
        output_dir: Directory to save merged wav
        end_extension_ms: Extra silence appended to the end (ms)

    Returns:
        A merged SegmentMetadata instance
    """
    if not segments:
        raise ValueError("segments must not be empty")

    # Helper function to get attributes from either SegmentMetadata or dict
    def get_attr(seg, attr_name, default=None):
        if isinstance(seg, dict):
            return seg.get(attr_name, default)
        return getattr(seg, attr_name, default)

    # ---------- concat notes ----------
    words: List[str] = []
    durs: List[float] = []
    pitches: List[int] = []
    types: List[int] = []

    for i, seg in enumerate(segments):
        if i > 0:
            prev_seg = segments[i - 1]
            gap_ms = (
                get_attr(seg, "start_time_ms", 0)
                - get_attr(prev_seg, "end_time_ms", 0)
            )
            if gap_ms > 0:
                words.append("<SP>")
                durs.append(gap_ms / 1000.0)
                pitches.append(0)
                types.append(1)

        words.extend(get_attr(seg, "note_text", []))
        durs.extend(get_attr(seg, "note_dur", []))
        pitches.extend(get_attr(seg, "note_pitch", []))
        types.extend(get_attr(seg, "note_type", []))

    if end_extension_ms > 0:
        words.append("<SP>")
        durs.append(end_extension_ms / 1000.0)
        pitches.append(0)
        types.append(1)

    # ---------- merge consecutive <SP> ----------
    merged_words, merged_durs, merged_pitches, merged_types = [], [], [], []
    for w, d, p, t in zip(words, durs, pitches, types):
        if merged_words and w == "<SP>" and merged_words[-1] == "<SP>":
            merged_durs[-1] += d
        else:
            merged_words.append(w)
            merged_durs.append(d)
            merged_pitches.append(p)
            merged_types.append(t)

    # ---------- dominant language ----------
    languages = [get_attr(s, "language", "Mandarin") for s in segments if get_attr(s, "language")]
    language = (
        max(languages, key=languages.count)
        if languages
        else "Mandarin"
    )

    # ---------- time & audio ----------
    start_ms = get_attr(segments[0], "start_time_ms", 0)
    end_ms = get_attr(segments[-1], "end_time_ms", 0) + end_extension_ms
    start_sample = start_ms * sample_rate // 1000
    end_sample = end_ms * sample_rate // 1000

    # ---------- naming ----------
    first_item_name = get_attr(segments[0], "item_name", "segment")
    song_prefix = "_".join(first_item_name.split("_")[:-1])
    item_name = f"{song_prefix}_{start_ms}_{end_ms}"

    wav_path = output_dir / f"{item_name}.wav"
    sf.write(
        wav_path,
        audio[start_sample:end_sample],
        sample_rate,
    )

    return SegmentMetadata(
        item_name=item_name,
        wav_fn=str(wav_path),
        language=language,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        note_text=merged_words,
        note_dur=merged_durs,
        note_pitch=merged_pitches,
        note_type=merged_types,
        origin_wav_fn=get_attr(segments[0], "origin_wav_fn", ""),
    )


def convert_metadata(item) -> Dict:
    """
    Convert internal SegmentMetadata into final json-serializable format.
    """
    f0_path = item.wav_fn.replace(".wav", "_f0.npy")
    f0 = np.load(f0_path)

    return {
        "index": item.item_name,
        "language": item.language,
        "time": [item.start_time_ms, item.end_time_ms],
        "duration": " ".join(f"{d:.2f}" for d in item.note_dur),
        "text": " ".join(item.note_text),
        "phoneme": " ".join(
            g2p_transform(item.note_text, item.language)
        ),
        "note_pitch": " ".join(map(str, item.note_pitch)),
        "note_type": " ".join(map(str, item.note_type)),
        "f0": " ".join(f"{x:.1f}" for x in f0),
    }


def merge_short_segments(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[SegmentMetadata],
    output_dir: str,
    max_gap_ms: int = 10000,
    max_duration_ms: int = 60000,
    end_extension_ms: int = 0,
) -> List[SegmentMetadata]:
    """
    Merge short segments into longer audio chunks.
    
    Args:
        audio: Full vocal audio waveform
        sample_rate: Audio sample rate
        segments: List of SegmentMetadata or dict objects
        output_dir: Directory to save merged segments
        max_gap_ms: Maximum gap between segments to merge (ms)
        max_duration_ms: Maximum duration of merged segment (ms)
        end_extension_ms: Extra silence to append at end (ms)
    
    Returns:
        List of merged SegmentMetadata objects
    """
    os.makedirs(output_dir, exist_ok=True)

    merged_segments = []
    current_group = []
    current_len = 0
    prev_end = -1

    for seg in segments:
        if isinstance(seg, dict):
            start_time = seg.get("start_time_ms", 0)
            end_time = seg.get("end_time_ms", 0)
        else:
            start_time = seg.start_time_ms
            end_time = seg.end_time_ms
        
        if (
            current_group
            and (start_time - prev_end > max_gap_ms
                 or current_len + end_time - start_time > max_duration_ms)
        ):
            merged_segments.append(_merge_group(
                audio, sample_rate, current_group, output_dir, end_extension_ms
            ))
            current_group = []
            current_len = 0

        current_group.append(seg)
        current_len += end_time - start_time
        prev_end = end_time

    if current_group:
        merged_segments.append(_merge_group(
            audio, sample_rate, current_group, output_dir, end_extension_ms
        ))

    return merged_segments
