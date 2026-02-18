"""
SoulX-Singer MIDI <-> metadata converter.

Converts between SoulX-Singer-style metadata JSON (with note_text, note_dur,
note_pitch, note_type per segment) and standard MIDI files. Uses an internal
Note dataclass (start_s, note_dur, note_text, note_pitch, note_type) as the
intermediate representation.
"""
import os
import json
import shutil
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import librosa
import mido
from soundfile import write

from .f0_extraction import F0Extractor
from .g2p import g2p_transform


# Audio and segmenting constants (used by _edit_data_to_meta)
SAMPLE_RATE = 44100
DEFAULT_LANGUAGE = "Mandarin"
MAX_GAP_SEC = 5.0  # gap (sec) above which we start a new segment
MAX_SEGMENT_DUR_SUM_SEC = 60.0  # max cumulative note duration per segment (sec)
MIN_GAP_THRESHOLD_SEC = 0.001  # ignore gaps smaller than this
LONG_SILENCE_THRESHOLD_SEC = 0.05  # treat as separate <SP> if gap larger
MAX_LEADING_SP_DUR_SEC = 2.0  # cap leading silence in a segment to this (sec)
DEFAULT_RMVPE_MODEL_PATH = "models/SoulX-Singer/preprocessors/rmvpe/rmvpe.pt"


@dataclass
class Note:
    """Single note: text, duration (seconds), pitch (MIDI), type. start_s is absolute start time in seconds (for ordering / MIDI)."""
    start_s: float
    note_dur: float
    note_text: str
    note_pitch: int
    note_type: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.note_dur



def remove_duplicate_segments(meta_data: List[dict]) -> None:
    """Merge consecutive identical notes (same text, pitch, type) within each segment. Mutates meta_data in place."""
    for idx, segment in enumerate(meta_data):
        texts = segment["note_text"]
        durs = segment["note_dur"]
        pitches = segment["note_pitch"]
        types = segment["note_type"]
        new_texts = []
        new_durs = []
        new_pitches = []
        new_types = []
        for i in range(len(texts)):
            if i == 0:
                new_texts.append(texts[i])
                new_durs.append(durs[i])
                new_pitches.append(pitches[i])
                new_types.append(types[i])
                continue
            t, d, p, ty = texts[i], durs[i], pitches[i], types[i]
            if t == "<SP>" and texts[i - 1] == "<SP>":
                new_durs[-1] += d
                continue
            if t == texts[i - 1] and p == pitches[i - 1] and ty == types[i - 1]:
                new_durs[-1] += d
            else:
                new_texts.append(t)
                new_durs.append(d)
                new_pitches.append(p)
                new_types.append(ty)
        meta_data[idx]["note_text"] = new_texts
        meta_data[idx]["note_dur"] = new_durs
        meta_data[idx]["note_pitch"] = new_pitches
        meta_data[idx]["note_type"] = new_types

def meta2notes(meta_path: str) -> List[Note]:
    """Parse SoulX-Singer metadata JSON into a flat list of Note (absolute start_s)."""
    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError(f"Metadata must be a list of segments, got {type(segments).__name__}")
    if not segments:
        raise ValueError("Metadata has no segments.")

    notes: List[Note] = []
    for seg in segments:
        offset_s = seg["time"][0] / 1000
        words = [str(x).replace("<AP>", "<SP>") for i, x in enumerate(seg["text"].split())]
        word_durs = [float(x) for x in seg["duration"].split()]
        pitches = [int(x) for x in seg["note_pitch"].split()]
        types = [int(x) if words[i] != "<SP>" else 1 for i, x in enumerate(seg["note_type"].split())]
        if len(words) != len(word_durs) or len(word_durs) != len(pitches) or len(pitches) != len(types):
            raise ValueError(
                f"Length mismatch in segment {seg.get('item_name', '?')}: "
                "note_text, note_dur, note_pitch, note_type must have same length"
            )
        current_s = offset_s
        for text, dur, pitch, type_ in zip(words, word_durs, pitches, types):
            notes.append(
                Note(
                    start_s=current_s,
                    note_dur=float(dur),
                    note_text=str(text),
                    note_pitch=int(pitch),
                    note_type=int(type_),
                )
            )
            current_s += float(dur)
    return notes

def _append_segment_to_meta(
    meta_path_str: str,
    cut_wavs_output_dir: str,
    vocal_file: str,
    audio_data: Any,
    meta_data: List[dict],
    note_start: List[float],
    note_end: List[float],
    note_text: List[Any],
    note_pitch: List[Any],
    note_type: List[Any],
    note_dur: List[float],
    end_time_ms_override: float | None = None,
) -> None:
    """Write one segment wav and append one segment dict to meta_data. Caller clears note_* lists after."""
    base_name = os.path.splitext(os.path.basename(meta_path_str))[0]
    item_name = f"{base_name}_{len(meta_data)}"
    wav_fn = os.path.join(cut_wavs_output_dir, f"{item_name}.wav")
    start_ms = int(note_start[0] * 1000)
    end_ms = (
        int(end_time_ms_override)
        if end_time_ms_override is not None
        else int(note_end[-1] * 1000)
    )
    start_sample = int(note_start[0] * SAMPLE_RATE)
    end_sample = int(note_end[-1] * SAMPLE_RATE)
    write(wav_fn, audio_data[start_sample:end_sample], SAMPLE_RATE)
    meta_data.append({
        "item_name": item_name,
        "wav_fn": wav_fn,
        "origin_wav_fn": vocal_file,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "language": DEFAULT_LANGUAGE,
        "note_text": list(note_text),
        "note_pitch": list(note_pitch),
        "note_type": list(note_type),
        "note_dur": list(note_dur),
    })


def convert_meta(meta_data: List[dict], rmvpe_model_path, device="cuda"):
    pitch_extractor = F0Extractor(rmvpe_model_path, device=device, verbose=False)
    converted_data = []

    for item in meta_data:
        wav_fn = item.get("wav_fn")
        if not wav_fn or not os.path.isfile(wav_fn):
            raise FileNotFoundError(f"Segment wav file not found: {wav_fn}")
        f0 = pitch_extractor.process(wav_fn)
        converted_item = {
            "index": item.get("item_name"),
            "language": item.get("language"),
            "time": [item.get("start_time_ms", 0), item.get("end_time_ms", sum(item["note_dur"]) * 1000)],
            "duration": " ".join(str(round(x, 2)) for x in item.get("note_dur", [])),
            "text": " ".join(item.get("note_text", [])),
            "phoneme": " ".join(g2p_transform(item.get("note_text", []), DEFAULT_LANGUAGE)),
            "note_pitch": " ".join(str(x) for x in item.get("note_pitch", [])),
            "note_type": " ".join(str(x) for x in item.get("note_type", [])),
            "f0": " ".join(str(round(float(x), 1)) for x in f0),
        }
        converted_data.append(converted_item)

    return converted_data


def _edit_data_to_meta(
    meta_path_str: str,
    edit_data: List[dict],
    vocal_file: str,
    rmvpe_model_path: str | None = None,
    device: str = "cuda",
) -> None:
    """Write SoulX-Singer metadata JSON from edit_data (list of {start, end, note_text, note_pitch, note_type})."""
    # Use a fixed temporary directory for cut wavs
    cut_wavs_output_dir = os.path.join(os.path.dirname(vocal_file), "cut_wavs_tmp")
    os.makedirs(cut_wavs_output_dir, exist_ok=True)

    note_text: List[Any] = []
    note_pitch: List[Any] = []
    note_type: List[Any] = []
    note_dur: List[float] = []
    note_start: List[float] = []
    note_end: List[float] = []
    prev_end = 0.0
    meta_data: List[dict] = []
    audio_data, _ = librosa.load(vocal_file, sr=SAMPLE_RATE, mono=True)
    dur_sum = 0.0

    for entry in edit_data:
        start = float(entry["start"])
        end = float(entry["end"])
        text = entry["note_text"]
        pitch = entry["note_pitch"]
        type_ = entry["note_type"]

        if text == "" or pitch == "" or type_ == "":
            note_text.append("<SP>")
            note_pitch.append(0)
            note_type.append(1)
            note_dur.append(end - start)
            note_start.append(start)
            note_end.append(end)
            prev_end = end
            dur_sum += end - start
            continue

        if (
            len(note_text) > 0
            and note_text[-1] == "<SP>"
            and note_dur[-1] > MAX_LEADING_SP_DUR_SEC
        ):
            cut_time = note_dur[-1] - MAX_LEADING_SP_DUR_SEC
            note_dur[-1] = MAX_LEADING_SP_DUR_SEC
            end_ms_override = note_end[-1] * 1000 - cut_time * 1000
            _append_segment_to_meta(
                meta_path_str,
                cut_wavs_output_dir,
                vocal_file,
                audio_data,
                meta_data,
                note_start,
                note_end,
                note_text,
                note_pitch,
                note_type,
                note_dur,
                end_time_ms_override=end_ms_override,
            )
            note_text = []
            note_pitch = []
            note_type = []
            note_dur = []
            note_start = []
            note_end = []
            prev_end = start
            dur_sum = 0.0

        gap_from_prev = start - prev_end
        gap_from_last_note = (start - note_end[-1]) if note_end else 0.0
        if (
            gap_from_prev >= MAX_GAP_SEC
            or gap_from_last_note >= MAX_GAP_SEC
            or dur_sum >= MAX_SEGMENT_DUR_SUM_SEC
        ):
            if len(note_text) > 0:
                _append_segment_to_meta(
                    meta_path_str,
                    cut_wavs_output_dir,
                    vocal_file,
                    audio_data,
                    meta_data,
                    note_start,
                    note_end,
                    note_text,
                    note_pitch,
                    note_type,
                    note_dur,
                )
                note_text = []
                note_pitch = []
                note_type = []
                note_dur = []
                note_start = []
                note_end = []
                prev_end = start
                dur_sum = 0.0

        if start - prev_end > MIN_GAP_THRESHOLD_SEC:
            if start - prev_end > LONG_SILENCE_THRESHOLD_SEC or len(note_text) == 0:
                note_text.append("<SP>")
                note_pitch.append(0)
                note_type.append(1)
                note_dur.append(start - prev_end)
                note_start.append(prev_end)
                note_end.append(start)
            else:
                if len(note_dur) > 0:
                    note_dur[-1] += start - prev_end
                    note_end[-1] = start

        prev_end = end
        note_text.append(text)
        note_pitch.append(int(pitch))
        note_type.append(int(type_))
        note_dur.append(end - start)
        note_start.append(start)
        note_end.append(end)
        dur_sum += end - start

    if len(note_text) > 0:
        _append_segment_to_meta(
            meta_path_str,
            cut_wavs_output_dir,
            vocal_file,
            audio_data,
            meta_data,
            note_start,
            note_end,
            note_text,
            note_pitch,
            note_type,
            note_dur,
        )

    remove_duplicate_segments(meta_data)

    _rmvpe_path = rmvpe_model_path or DEFAULT_RMVPE_MODEL_PATH
    converted_data = convert_meta(meta_data, _rmvpe_path, device)

    with open(meta_path_str, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    # Clean up temporary cut wavs directory
    try:
        shutil.rmtree(cut_wavs_output_dir, ignore_errors=True)
    except Exception:
        pass


def notes2meta(
    notes: List[Note],
    meta_path: str,
    vocal_file: str,
    rmvpe_model_path: str | None = None,
    device: str = "cuda",
) -> None:
    """Write SoulX-Singer metadata JSON from a list of Note (segmenting + wav cuts)."""
    edit_data = [
        {
            "start": n.start_s,
            "end": n.end_s,
            "note_text": n.note_text,
            "note_pitch": str(n.note_pitch),
            "note_type": str(n.note_type),
        }
        for n in notes
    ]
    _edit_data_to_meta(
        str(meta_path),
        edit_data,
        vocal_file,
        rmvpe_model_path=rmvpe_model_path,
        device=device,
    )


@dataclass(frozen=True)
class MidiDefaults:
    ticks_per_beat: int = 500
    tempo: int = 500000  # microseconds per beat (120 BPM)
    time_signature: Tuple[int, int] = (4, 4)
    velocity: int = 64


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int) -> int:
    return int(round(seconds * ticks_per_beat * 1_000_000 / tempo))


def notes2midi(
    notes: List[Note],
    midi_path: str,
    defaults: MidiDefaults | None = None,
) -> None:
    """Write MIDI file from a list of Note."""
    defaults = defaults or MidiDefaults()
    if not notes:
        raise ValueError("Empty note list.")

    events: List[Tuple[int, int, Union[mido.Message, mido.MetaMessage]]] = []
    for n in notes:
        start_s = n.start_s
        end_s = n.end_s
        if end_s <= start_s:
            continue

        start_ticks = _seconds_to_ticks(
            start_s, defaults.ticks_per_beat, defaults.tempo
        )
        end_ticks = _seconds_to_ticks(
            end_s, defaults.ticks_per_beat, defaults.tempo
        )
        if end_ticks <= start_ticks:
            end_ticks = start_ticks + 1

        lyric = n.note_text
        try:
            lyric = lyric.encode("utf-8").decode("latin1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        if n.note_type == 3:
            lyric = "-"

        events.append(
            (start_ticks, 1, mido.MetaMessage("lyrics", text=lyric, time=0))
        )
        events.append(
            (
                start_ticks,
                2,
                mido.Message(
                    "note_on",
                    note=n.note_pitch,
                    velocity=defaults.velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_ticks,
                0,
                mido.Message("note_off", note=n.note_pitch, velocity=0, time=0),
            )
        )

    events.sort(key=lambda x: (x[0], x[1]))

    mid = mido.MidiFile(ticks_per_beat=defaults.ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=defaults.tempo, time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=defaults.time_signature[0],
            denominator=defaults.time_signature[1],
            time=0,
        )
    )

    last_tick = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last_tick)
        track.append(msg)
        last_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(midi_path)


def midi2notes(midi_path: str) -> List[Note]:
    """Parse MIDI file into a list of Note. Merges all tracks; tempo from last set_tempo event."""
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000

    raw_notes: List[dict] = []
    lyrics: List[Tuple[int, str]] = []

    for track in mid.tracks:
        abs_ticks = 0
        active = {}
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "lyrics":
                text = msg.text
                try:
                    text = text.encode("latin1").decode("utf-8")
                except Exception:
                    pass
                lyrics.append((abs_ticks, text))
            elif msg.type == "note_on":
                key = (msg.channel, msg.note)
                if msg.velocity > 0:
                    active[key] = (abs_ticks, msg.velocity)
                else:
                    if key in active:
                        start_ticks, vel = active.pop(key)
                        raw_notes.append(
                            {
                                "midi": msg.note,
                                "start_ticks": start_ticks,
                                "duration_ticks": abs_ticks - start_ticks,
                                "velocity": vel,
                                "lyric": "",
                            }
                        )
            elif msg.type == "note_off":
                key = (msg.channel, msg.note)
                if key in active:
                    start_ticks, vel = active.pop(key)
                    raw_notes.append(
                        {
                            "midi": msg.note,
                            "start_ticks": start_ticks,
                            "duration_ticks": abs_ticks - start_ticks,
                            "velocity": vel,
                            "lyric": "",
                        }
                    )

    if not raw_notes:
        raise ValueError("No notes found in MIDI file")

    for n in raw_notes:
        n["end_ticks"] = n["start_ticks"] + n["duration_ticks"]

    raw_notes.sort(key=lambda n: n["start_ticks"])
    lyrics.sort(key=lambda x: x[0])

    trimmed = []
    for note in raw_notes:
        while trimmed:
            prev = trimmed[-1]
            if note["start_ticks"] < prev["end_ticks"]:
                prev["end_ticks"] = note["start_ticks"]
                prev["duration_ticks"] = prev["end_ticks"] - prev["start_ticks"]
                if prev["duration_ticks"] <= 0:
                    trimmed.pop()
                    continue
            break
        trimmed.append(note)
    raw_notes = trimmed

    tolerance = ticks_per_beat // 100
    lyric_idx = 0
    for note in raw_notes:
        while lyric_idx < len(lyrics) and lyrics[lyric_idx][0] < note["start_ticks"] - tolerance:
            lyric_idx += 1
        if lyric_idx < len(lyrics):
            lyric_ticks, lyric_text = lyrics[lyric_idx]
            if abs(lyric_ticks - note["start_ticks"]) <= tolerance:
                note["lyric"] = lyric_text
                lyric_idx += 1

    def ticks_to_seconds(ticks: int) -> float:
        return (ticks / ticks_per_beat) * (tempo / 1_000_000)

    result: List[Note] = []
    prev_end_s = 0.0
    for idx, n in enumerate(raw_notes):
        start_s = ticks_to_seconds(n["start_ticks"])
        end_s = ticks_to_seconds(n["end_ticks"])
        if prev_end_s > start_s:
            start_s = prev_end_s
        dur_s = end_s - start_s
        if dur_s <= 0:
            continue

        lyric = n.get("lyric", "")
        if not lyric:
            tp = 2
            text = "啦"
        elif lyric == "<SP>":
            tp = 1
            text = "<SP>"
        elif lyric == "-":
            tp = 3
            text = raw_notes[idx - 1].get("lyric", "-") if idx > 0 else "-"
        else:
            tp = 2
            text = lyric

        result.append(
            Note(
                start_s=start_s,
                note_dur=dur_s,
                note_text=text,
                note_pitch=n["midi"],
                note_type=tp,
            )
        )
        prev_end_s = end_s

    return result


def meta2midi(meta_path: str, midi_path: str, defaults: MidiDefaults | None = None) -> None:
    """Convert SoulX-Singer metadata JSON to MIDI file (meta -> List[Note] -> midi)."""
    notes = meta2notes(meta_path)
    notes2midi(notes, midi_path, defaults)
    print(f"Saved MIDI to {midi_path}")


def midi2meta(
    midi_path: str,
    meta_path: str,
    vocal_file: str,
    rmvpe_model_path: str | None = None,
    device: str = "cuda",
) -> None:
    """Convert MIDI file to SoulX-Singer metadata JSON (midi -> List[Note] -> meta)."""
    meta_dir = os.path.dirname(meta_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)
    # cut_wavs will be written to a fixed temporary directory inside _edit_data_to_meta
    notes = midi2notes(midi_path)
    notes2meta(
        notes,
        meta_path,
        vocal_file,
        rmvpe_model_path=rmvpe_model_path,
        device=device,
    )
    print(f"Saved Meta to {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SoulX-Singer metadata JSON <-> MIDI."
    )
    parser.add_argument("--meta", type=str, help="Path to metadata JSON")
    parser.add_argument("--midi", type=str, help="Path to MIDI file")
    parser.add_argument("--vocal", type=str, help="Path to vocal wav (for midi2meta)")
    parser.add_argument(
        "--meta2midi",
        action="store_true",
        help="Convert meta -> midi (requires --meta and --midi)",
    )
    parser.add_argument(
        "--midi2meta",
        action="store_true",
        help="Convert midi -> meta (requires --midi, --meta, --vocal, --cut_wavs_dir)",
    )
    parser.add_argument(
        "--rmvpe_model_path",
        type=str,
        help="Path to RMVPE model",
        default="models/SoulX-Singer/preprocessors/rmvpe/rmvpe.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for RMVPE",
        default="cuda",
    )
    args = parser.parse_args()

    if args.meta2midi:
        if not args.meta or not args.midi:
            parser.error("--meta2midi requires --meta and --midi")
        meta2midi(args.meta, args.midi)
    elif args.midi2meta:
        if not args.midi or not args.meta or not args.vocal:
            parser.error(
                "--midi2meta requires --midi, --meta, --vocal"
            )
        midi2meta(
            args.midi,
            args.meta,
            args.vocal,
            rmvpe_model_path=args.rmvpe_model_path,
            device=args.device,
        )
    else:
        parser.print_help()