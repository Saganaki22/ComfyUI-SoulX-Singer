import { nanoid } from 'nanoid'
import { create } from 'zustand'
import type { NoteEvent, TimeSignature } from '../types'

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max)

export type MidiStore = {
  tempo: number
  timeSignature: TimeSignature
  notes: NoteEvent[]
  selectedId: string | null
  playhead: number
  ppq: number | undefined  // Ticks per quarter note (for preserving original MIDI timing)
  addNote: (partial?: Partial<NoteEvent>) => NoteEvent
  updateNote: (id: string, partial: Partial<NoteEvent>) => void
  removeNote: (id: string) => void
  setNotes: (notes: NoteEvent[]) => void
  setTempo: (tempo: number) => void
  setTimeSignature: (sig: TimeSignature) => void
  setPpq: (ppq: number | undefined) => void
  select: (id: string | null) => void
  setLyric: (id: string, lyric: string) => void
  setPlayhead: (beat: number) => void
  clear: () => void
}

const defaultNotes: NoteEvent[] = [
  { id: nanoid(), midi: 64, start: 0, duration: 1.5, velocity: 0.9, lyric: 'la' },
  { id: nanoid(), midi: 67, start: 1.5, duration: 1.5, velocity: 0.85, lyric: 'na' },
  { id: nanoid(), midi: 69, start: 3, duration: 2, velocity: 0.8, lyric: 'ah' },
]

export const useMidiStore = create<MidiStore>((set) => ({
  tempo: 110,
  timeSignature: [4, 4],
  notes: defaultNotes,
  selectedId: null,
  playhead: 0,
  ppq: undefined,
  addNote: (partial = {}) => {
    const note: NoteEvent = {
      id: nanoid(),
      midi: partial.midi ?? 64,
      start: partial.start ?? 0,
      duration: partial.duration ?? 1,
      velocity: clamp(partial.velocity ?? 0.85, 0, 1),
      lyric: partial.lyric ?? '',
    }
    set((state) => ({ notes: [...state.notes, note] }))
    return note
  },
  updateNote: (id, partial) => {
    set((state) => ({
      notes: state.notes.map((note) =>
        note.id === id
          ? {
              ...note,
              ...partial,
              duration: Math.max(partial.duration ?? note.duration, 0.0625),
            }
          : note,
      ),
    }))
  },
  removeNote: (id) => set((state) => ({ notes: state.notes.filter((n) => n.id !== id) })),
  setNotes: (notes) => set(() => ({ notes })),
  setTempo: (tempo) => set(() => ({ tempo: clamp(tempo, 30, 240) })),
  setTimeSignature: (sig) => set(() => ({ timeSignature: sig })),
  setPpq: (ppq) => set(() => ({ ppq })),
  select: (id) => set(() => ({ selectedId: id })),
  setLyric: (id, lyric) =>
    set((state) => ({
      notes: state.notes.map((note) => (note.id === id ? { ...note, lyric } : note)),
    })),
  setPlayhead: (beat) => set(() => ({ playhead: Math.max(beat, 0) })),
  clear: () => set(() => ({ notes: [], selectedId: null })),
}))
