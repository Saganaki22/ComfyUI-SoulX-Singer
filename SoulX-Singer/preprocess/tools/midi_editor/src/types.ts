export type NoteEvent = {
  id: string
  midi: number
  start: number // in beats
  duration: number // in beats
  velocity: number
  lyric: string
}

export type TimeSignature = [number, number]

export type ProjectSnapshot = {
  tempo: number
  timeSignature: TimeSignature
  notes: NoteEvent[]
  ppq?: number  // Ticks per quarter note (for preserving original MIDI timing)
}
