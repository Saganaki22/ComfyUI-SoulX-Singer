import { useEffect, useMemo, useRef, useState, useCallback, memo } from 'react'
import type React from 'react'
import type { NoteEvent, TimeSignature } from '../types'
import { PITCH_WIDTH, LOW_NOTE, HIGH_NOTE } from '../constants'

const midiToName = (midi: number) => {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const octave = Math.floor(midi / 12) - 1
  return `${names[midi % 12]}${octave}`
}

// Memoized note component to prevent unnecessary re-renders
const NoteChip = memo(function NoteChip({
  note,
  left,
  top,
  width,
  height,
  fontSize,
  isSelected,
  isOverlapping,
  onPointerDown,
  onDoubleClick,
}: {
  note: NoteEvent
  left: number
  top: number
  width: number
  height: number
  fontSize: number
  isSelected: boolean
  isOverlapping: boolean
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>, mode: 'move' | 'resize-start' | 'resize-end') => void
  onDoubleClick: (event: React.MouseEvent<HTMLDivElement>) => void
}) {
  return (
    <div
      className={`note-chip ${isSelected ? 'note-active' : ''} ${isOverlapping ? 'note-overlap' : ''}`}
      style={{ 
        left, 
        top: top + 1,
        width, 
        height,
        willChange: 'transform', // GPU acceleration hint
      }}
      onPointerDown={(e) => onPointerDown(e, 'move')}
      onDoubleClick={onDoubleClick}
    >
      <div className="note-label" style={{ fontSize }}>
        <span>{note.lyric || '\u00a0'}</span>
      </div>
      <div className="note-handle start" onPointerDown={(e) => { e.stopPropagation(); onPointerDown(e, 'resize-start') }} />
      <div className="note-handle end" onPointerDown={(e) => { e.stopPropagation(); onPointerDown(e, 'resize-end') }} />
    </div>
  )
})

// Dynamic snap based on zoom level - higher zoom = finer snap
const getSnapSeconds = (gridSecondWidth: number) => {
  // At base width (80px/s), snap is 0.1s
  // At 2x zoom (160px/s), snap is 0.05s
  // At 4x zoom (320px/s), snap is 0.025s
  // At 8x zoom (640px/s), snap is 0.01s
  const baseSnap = 0.1
  const zoomFactor = gridSecondWidth / 80
  return Math.max(0.01, baseSnap / zoomFactor)
}

const snapSeconds = (value: number, gridSecondWidth: number) => {
  const snap = getSnapSeconds(gridSecondWidth)
  return Math.max(0, Math.round(value / snap) * snap)
}

export type PianoRollProps = {
  notes: NoteEvent[]
  selectedId: string | null
  timeSignature: TimeSignature
  tempo: number
  playhead: number // in beats
  selectionStart: number | null // in seconds
  selectionEnd: number | null // in seconds
  onAddNote: (note: Partial<NoteEvent>) => NoteEvent
  onUpdateNote: (id: string, patch: Partial<NoteEvent>) => void
  onSelect: (id: string | null) => void
  onSeek: (beat: number) => void
  onScroll?: (left: number) => void
  onZoom?: (deltaH: number, deltaV: number) => void
  onPlayNote?: (midi: number) => void
  onFocusLyric?: (noteId: string) => void
  onSelectionChange?: (start: number | null, end: number | null) => void
  isSelectingRange?: boolean
  audioDuration?: number
  gridSecondWidth: number
  rowHeight: number
}

export function PianoRoll({
  notes,
  selectedId,
  timeSignature: _timeSignature,
  tempo,
  playhead,
  selectionStart,
  selectionEnd,
  onAddNote,
  onSelect,
  onUpdateNote,
  onSeek,
  onScroll,
  onZoom,
  onPlayNote,
  onFocusLyric,
  onSelectionChange,
  isSelectingRange = false,
  audioDuration = 0,
  gridSecondWidth,
  rowHeight
}: PianoRollProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)
  const rulerScrollRef = useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [scrollLeft, setScrollLeft] = useState(0)
  const [viewportWidth, setViewportWidth] = useState(800)
  const [viewportHeight, setViewportHeight] = useState(400)
  const dragRef = useRef<{
    id: string
    mode: 'move' | 'resize-start' | 'resize-end'
    originX: number
    originY: number
    startSeconds: number
    durationSeconds: number
    midi: number
    lastMidi?: number // Track last midi for pitch change sound
  } | null>(null)
  
  // Selection drag state
  const selectionDragRef = useRef<{
    startX: number
    startSeconds: number
  } | null>(null)
  
  // Store callbacks in refs to avoid stale closures in event handlers
  const onPlayNoteRef = useRef(onPlayNote)
  const onUpdateNoteRef = useRef(onUpdateNote)
  
  useEffect(() => {
    onPlayNoteRef.current = onPlayNote
    onUpdateNoteRef.current = onUpdateNote
  }, [onPlayNote, onUpdateNote])

  // Conversion helpers
  const beatToSeconds = useCallback((beat: number) => beat * (60 / tempo), [tempo])
  const secondsToBeat = useCallback((seconds: number) => seconds / (60 / tempo), [tempo])

  // Calculate dimensions
  const totalRows = HIGH_NOTE - LOW_NOTE + 1
  const contentHeight = totalRows * rowHeight
  const [containerWidth, setContainerWidth] = useState(1200)
  
  // Track container size
  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return
    
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
        setViewportWidth(entry.contentRect.width)
        setViewportHeight(entry.contentRect.height)
      }
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])
  
  const maxSeconds = useMemo(() => {
    const noteEndSeconds = notes.reduce((acc, n) => {
      const endBeat = n.start + n.duration
      return Math.max(acc, beatToSeconds(endBeat))
    }, 8)
    // Ensure grid extends at least 2x the visible area for smoother scrolling
    const minSecondsForView = (containerWidth / gridSecondWidth) * 2
    return Math.max(noteEndSeconds + 10, audioDuration + 10, minSecondsForView, 30)
  }, [notes, audioDuration, beatToSeconds, containerWidth, gridSecondWidth])

  const contentWidth = maxSeconds * gridSecondWidth

  // Drag handlers - use refs to avoid stale closure issues
  const handlePointerMove = useCallback((event: PointerEvent) => {
    const drag = dragRef.current
    if (!drag) return

    const dxSeconds = (event.clientX - drag.originX) / gridSecondWidth
    const dy = (event.clientY - drag.originY) / rowHeight

    if (drag.mode === 'move') {
      const nextSeconds = snapSeconds(drag.startSeconds + dxSeconds, gridSecondWidth)
      const nextMidi = Math.min(HIGH_NOTE, Math.max(LOW_NOTE, Math.round(drag.midi - dy)))
      
      // Play sound when pitch changes
      if (nextMidi !== drag.lastMidi && onPlayNoteRef.current) {
        onPlayNoteRef.current(nextMidi)
        drag.lastMidi = nextMidi
      }
      
      onUpdateNoteRef.current(drag.id, { 
        start: secondsToBeat(nextSeconds), 
        midi: nextMidi 
      })
    }

    if (drag.mode === 'resize-start') {
      const nextSeconds = snapSeconds(drag.startSeconds + dxSeconds, gridSecondWidth)
      const delta = drag.startSeconds - nextSeconds
      const nextDurationSeconds = Math.max(0.05, drag.durationSeconds + delta)
      onUpdateNoteRef.current(drag.id, { 
        start: secondsToBeat(nextSeconds), 
        duration: secondsToBeat(nextDurationSeconds) 
      })
    }

    if (drag.mode === 'resize-end') {
      const nextDurationSeconds = Math.max(0.05, snapSeconds(drag.durationSeconds + dxSeconds, gridSecondWidth))
      onUpdateNoteRef.current(drag.id, { duration: secondsToBeat(nextDurationSeconds) })
    }
  }, [gridSecondWidth, rowHeight, secondsToBeat])

  const handlePointerUp = useCallback(() => {
    dragRef.current = null
    window.removeEventListener('pointermove', handlePointerMove)
    window.removeEventListener('pointerup', handlePointerUp)
  }, [handlePointerMove])

  useEffect(() => {
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
    }
  }, [handlePointerMove, handlePointerUp])

  // Scroll sync
  useEffect(() => {
    const container = scrollContainerRef.current
    const ruler = rulerScrollRef.current
    if (!container || !ruler) return

    const handleScroll = () => {
      ruler.scrollLeft = container.scrollLeft
      setScrollTop(container.scrollTop)
      setScrollLeft(container.scrollLeft)
      if (onScroll) onScroll(container.scrollLeft)
    }

    container.addEventListener('scroll', handleScroll)
    return () => container.removeEventListener('scroll', handleScroll)
  }, [onScroll])

  // Zoom support via wheel/trackpad
  // Mac: Cmd+滚轮 (水平缩放), Cmd+Shift+滚轮 (垂直缩放), 或双指捏合
  // Windows/Linux: Ctrl+滚轮 (水平缩放), Ctrl+Shift+滚轮 (垂直缩放)
  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container || !onZoom) return

    const handleWheel = (e: WheelEvent) => {
      // Ctrl (Windows/Linux/捏合) or Cmd (Mac) triggers zoom
      const isZoomTrigger = e.ctrlKey || e.metaKey
      
      if (isZoomTrigger) {
        e.preventDefault()
        e.stopPropagation()
        
        // Use deltaY for zoom amount, normalize for different input methods
        // Pinch gestures typically have smaller delta values
        let delta = -e.deltaY
        if (Math.abs(delta) > 10) {
          // Likely a mouse wheel, scale down
          delta = delta * 0.01
        } else {
          // Likely a trackpad pinch, scale appropriately
          delta = delta * 0.05
        }
        
        // Shift or Alt/Option for vertical zoom, otherwise horizontal
        if (e.shiftKey || e.altKey) {
          onZoom(0, delta)
        } else {
          onZoom(delta, 0)
        }
      }
    }

    container.addEventListener('wheel', handleWheel, { passive: false })
    return () => container.removeEventListener('wheel', handleWheel)
  }, [onZoom])

  // Playhead auto-scroll
  useEffect(() => {
    if (!scrollContainerRef.current) return
    const container = scrollContainerRef.current
    const playheadX = beatToSeconds(playhead) * gridSecondWidth
    const viewStart = container.scrollLeft
    const viewEnd = viewStart + container.clientWidth

    if (playheadX > viewEnd) {
      container.scrollLeft = playheadX
    } else if (playheadX < viewStart) {
      container.scrollLeft = playheadX
    }
  }, [playhead, gridSecondWidth, beatToSeconds])

  // Selection auto-scroll
  useEffect(() => {
    if (!scrollContainerRef.current || !selectedId) return
    const note = notes.find((n) => n.id === selectedId)
    if (!note) return
    const container = scrollContainerRef.current
    const noteX = beatToSeconds(note.start) * gridSecondWidth
    const noteY = (HIGH_NOTE - note.midi) * rowHeight
    
    const viewStart = container.scrollLeft
    const viewEnd = viewStart + container.clientWidth
    if (noteX < viewStart + 50 || noteX > viewEnd - 50) {
      container.scrollLeft = Math.max(0, noteX - container.clientWidth * 0.35)
    }
    
    const viewTop = container.scrollTop
    const viewBottom = viewTop + container.clientHeight
    if (noteY < viewTop || noteY > viewBottom - rowHeight) {
      container.scrollTop = Math.max(0, noteY - container.clientHeight * 0.4)
    }
  }, [selectedId, notes, gridSecondWidth, rowHeight, beatToSeconds])

  const handleGridDoubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    // Only add note if clicking on empty space (not on a note)
    const target = event.target as HTMLElement
    if (target.closest('.note-chip')) return
    
    if (!scrollContainerRef.current) return
    const container = scrollContainerRef.current
    const rect = container.getBoundingClientRect()
    const x = event.clientX - rect.left + container.scrollLeft
    const y = event.clientY - rect.top + container.scrollTop
    
    const seconds = snapSeconds(x / gridSecondWidth, gridSecondWidth)
    const pitch = Math.min(HIGH_NOTE, Math.max(LOW_NOTE, HIGH_NOTE - Math.floor(y / rowHeight)))
    
    const created = onAddNote({ 
      start: secondsToBeat(seconds), 
      midi: pitch, 
      duration: secondsToBeat(0.5), 
      lyric: '' 
    })
    onSelect(created.id)
  }

  const startDrag = (
    event: React.PointerEvent<HTMLDivElement>,
    note: NoteEvent,
    mode: 'move' | 'resize-start' | 'resize-end',
  ) => {
    event.preventDefault()
    event.stopPropagation()
    dragRef.current = {
      id: note.id,
      mode,
      originX: event.clientX,
      originY: event.clientY,
      startSeconds: beatToSeconds(note.start),
      durationSeconds: beatToSeconds(note.duration),
      midi: note.midi,
      lastMidi: note.midi, // Initialize last midi
    }
    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    onSelect(note.id)
    
    // Play sound when clicking/selecting note
    if (onPlayNote) {
      onPlayNote(note.midi)
    }
  }

  // Second-based ruler labels
  const secondLabels = useMemo(() => {
    const labels = [] as Array<{ left: number; label: string }>
    const totalSeconds = Math.ceil(maxSeconds)
    for (let s = 0; s <= totalSeconds; s += 1) {
      labels.push({ left: s * gridSecondWidth, label: `${s}s` })
    }
    return labels
  }, [maxSeconds, gridSecondWidth])

  // Piano keys
  const pitchRows = useMemo(() => {
    const rows = [] as Array<{ midi: number; isBlack: boolean; label: string; isC: boolean }>
    const black = new Set([1, 3, 6, 8, 10])
    for (let p = HIGH_NOTE; p >= LOW_NOTE; p -= 1) {
      const name = midiToName(p)
      const isC = p % 12 === 0
      rows.push({ midi: p, isBlack: black.has(p % 12), label: name, isC })
    }
    return rows
  }, [])

  // Detect overlapping notes using optimized sweep line algorithm
  const overlappingNoteIds = useMemo(() => {
    if (notes.length < 2) return new Set<string>()
    
    const overlapping = new Set<string>()
    const sortedNotes = [...notes].sort((a, b) => a.start - b.start)
    const EPSILON = 0.05 // Tolerance for floating point comparison
    
    // Use a sliding window approach - more efficient for typical music data
    // Active notes: notes that haven't ended yet
    const activeNotes: typeof sortedNotes = []
    
    for (const note of sortedNotes) {
      // Remove notes that have ended before current note starts
      while (activeNotes.length > 0) {
        const firstActive = activeNotes[0]
        const firstActiveEnd = firstActive.start + firstActive.duration
        if (firstActiveEnd <= note.start + EPSILON) {
          activeNotes.shift()
        } else {
          break
        }
      }
      
      // Check overlap with remaining active notes
      for (const activeNote of activeNotes) {
        const activeEnd = activeNote.start + activeNote.duration
        if (note.start < activeEnd - EPSILON) {
          overlapping.add(activeNote.id)
          overlapping.add(note.id)
        }
      }
      
      // Add current note to active set (maintain sorted order by end time)
      const noteEnd = note.start + note.duration
      let insertIndex = activeNotes.length
      for (let i = 0; i < activeNotes.length; i++) {
        const aEnd = activeNotes[i].start + activeNotes[i].duration
        if (noteEnd < aEnd) {
          insertIndex = i
          break
        }
      }
      activeNotes.splice(insertIndex, 0, note)
    }
    return overlapping
  }, [notes])
  
  // Calculate visible area with buffer for smooth scrolling
  const BUFFER_PX = 200 // Render notes slightly outside viewport for smooth scrolling
  const visibleArea = useMemo(() => {
    return {
      left: Math.max(0, scrollLeft - BUFFER_PX),
      right: scrollLeft + viewportWidth + BUFFER_PX,
      top: Math.max(0, scrollTop - BUFFER_PX),
      bottom: scrollTop + viewportHeight + BUFFER_PX,
    }
  }, [scrollLeft, scrollTop, viewportWidth, viewportHeight])
  
  // Filter notes to only render visible ones (virtualization)
  const visibleNotes = useMemo(() => {
    return notes.filter(note => {
      const noteSeconds = beatToSeconds(note.start)
      const noteDurationSeconds = beatToSeconds(note.duration)
      const noteLeft = noteSeconds * gridSecondWidth
      const noteRight = noteLeft + noteDurationSeconds * gridSecondWidth
      const noteTop = (HIGH_NOTE - note.midi) * rowHeight
      const noteBottom = noteTop + rowHeight
      
      // Check if note intersects with visible area
      const horizontallyVisible = noteRight >= visibleArea.left && noteLeft <= visibleArea.right
      const verticallyVisible = noteBottom >= visibleArea.top && noteTop <= visibleArea.bottom
      
      return horizontallyVisible && verticallyVisible
    })
  }, [notes, visibleArea, gridSecondWidth, rowHeight, beatToSeconds])
  
  // Calculate visible grid lines (virtualization)
  const visibleGridLines = useMemo(() => {
    const startSecond = Math.max(0, Math.floor(visibleArea.left / gridSecondWidth) - 1)
    const endSecond = Math.ceil(visibleArea.right / gridSecondWidth) + 1
    const startRow = Math.max(0, Math.floor(visibleArea.top / rowHeight) - 1)
    const endRow = Math.min(totalRows, Math.ceil(visibleArea.bottom / rowHeight) + 1)
    
    return {
      horizontalLines: Array.from({ length: endRow - startRow + 1 }, (_, i) => startRow + i),
      verticalLines: Array.from({ length: endSecond - startSecond + 1 }, (_, i) => startSecond + i),
    }
  }, [visibleArea, gridSecondWidth, rowHeight, totalRows])

  const playheadSeconds = beatToSeconds(playhead)

  // Selection drag handlers
  const handleRulerPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isSelectingRange) {
      // Normal click to seek
      const rect = event.currentTarget.getBoundingClientRect()
      const x = event.clientX - rect.left + (rulerScrollRef.current?.scrollLeft ?? 0)
      const seconds = x / gridSecondWidth
      onSeek(secondsToBeat(seconds))
      return
    }
    
    // Start selection drag
    const rect = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - rect.left + (rulerScrollRef.current?.scrollLeft ?? 0)
    const seconds = Math.max(0, x / gridSecondWidth)
    
    selectionDragRef.current = {
      startX: event.clientX,
      startSeconds: seconds,
    }
    
    onSelectionChange?.(seconds, seconds)
    
    const handleSelectionMove = (e: PointerEvent) => {
      if (!selectionDragRef.current) return
      const currentX = e.clientX - rect.left + (rulerScrollRef.current?.scrollLeft ?? 0)
      const currentSeconds = Math.max(0, currentX / gridSecondWidth)
      const start = Math.min(selectionDragRef.current.startSeconds, currentSeconds)
      const end = Math.max(selectionDragRef.current.startSeconds, currentSeconds)
      onSelectionChange?.(start, end)
    }
    
    const handleSelectionUp = () => {
      selectionDragRef.current = null
      window.removeEventListener('pointermove', handleSelectionMove)
      window.removeEventListener('pointerup', handleSelectionUp)
    }
    
    window.addEventListener('pointermove', handleSelectionMove)
    window.addEventListener('pointerup', handleSelectionUp)
  }

  return (
    <div className="piano-shell">
      {/* Ruler */}
      <div className="ruler-shell">
        <div className="ruler-spacer" style={{ width: PITCH_WIDTH, flexShrink: 0 }} />
        <div
          ref={rulerScrollRef}
          className={`ruler-scroll ${isSelectingRange ? 'selecting' : ''}`}
          onPointerDown={handleRulerPointerDown}
        >
          <div className="ruler" style={{ width: contentWidth }}>
            {secondLabels.map((mark) => (
              <div key={mark.left} className="measure-mark" style={{ left: mark.left }}>
                <span>{mark.label}</span>
              </div>
            ))}
            {/* Selection range indicator */}
            {selectionStart !== null && selectionEnd !== null && selectionEnd > selectionStart && (
              <div 
                className="selection-range"
                style={{ 
                  left: selectionStart * gridSecondWidth,
                  width: (selectionEnd - selectionStart) * gridSecondWidth
                }}
              />
            )}
            {/* Ruler playhead indicator */}
            <div 
              className="ruler-playhead"
              style={{ left: playheadSeconds * gridSecondWidth }}
            />
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div className="roll-body">
        {/* Piano keys - synced with vertical scroll */}
        <div className="pitch-rail" style={{ width: PITCH_WIDTH }}>
          <div 
            className="pitch-rail-inner"
            style={{ 
              transform: `translateY(${-scrollTop}px)`,
              height: contentHeight
            }}
          >
            {pitchRows.map((pitch) => (
              <div
                key={pitch.midi}
                className={`pitch-cell ${pitch.isBlack ? 'pitch-black' : 'pitch-white'} ${pitch.isC ? 'pitch-c' : ''}`}
                style={{ height: rowHeight, cursor: 'pointer' }}
                onClick={() => onPlayNote?.(pitch.midi)}
                onMouseDown={(e) => e.preventDefault()}
              >
                <span className="pitch-label">{pitch.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Scrollable grid area */}
        <div
          ref={scrollContainerRef}
          className="roll-grid"
          onDoubleClick={handleGridDoubleClick}
        >
          <div 
            className="grid-content" 
            style={{ 
              width: contentWidth, 
              height: contentHeight, 
              position: 'relative' 
            }}
          >
            {/* SVG Grid - virtualized for performance */}
            <svg 
              className="grid-svg"
              width={contentWidth} 
              height={contentHeight}
              style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
            >
              {/* Horizontal lines (pitch rows) - only visible ones */}
              {visibleGridLines.horizontalLines.map(i => (
                <line
                  key={`h-${i}`}
                  x1={visibleArea.left}
                  y1={i * rowHeight}
                  x2={visibleArea.right}
                  y2={i * rowHeight}
                  stroke="var(--grid-line-minor)"
                  strokeWidth={1}
                />
              ))}
              {/* Vertical lines (seconds) - only visible ones */}
              {visibleGridLines.verticalLines.map(i => (
                <line
                  key={`v-${i}`}
                  x1={i * gridSecondWidth}
                  y1={visibleArea.top}
                  x2={i * gridSecondWidth}
                  y2={visibleArea.bottom}
                  stroke="var(--grid-line-minor)"
                  strokeWidth={1}
                />
              ))}
            </svg>

            {/* Selection range in grid */}
            {selectionStart !== null && selectionEnd !== null && selectionEnd > selectionStart && (
              <div 
                className="grid-selection-range"
                style={{ 
                  left: selectionStart * gridSecondWidth,
                  width: (selectionEnd - selectionStart) * gridSecondWidth,
                  height: contentHeight
                }}
              />
            )}

            {/* Playhead */}
            <div 
              className="playhead" 
              style={{ 
                left: playheadSeconds * gridSecondWidth, 
                height: contentHeight 
              }} 
            />

            {/* Notes - virtualized: only render visible notes */}
            {visibleNotes.map((note) => {
              const noteSeconds = beatToSeconds(note.start)
              const noteDurationSeconds = beatToSeconds(note.duration)
              const left = noteSeconds * gridSecondWidth
              const top = (HIGH_NOTE - note.midi) * rowHeight
              const noteWidthPx = Math.max(noteDurationSeconds * gridSecondWidth, 4)
              const noteHeight = rowHeight - 2
              const isOverlapping = overlappingNoteIds.has(note.id)
              // Dynamic font size based on row height (base: 12px at 20px row height)
              const fontSize = Math.max(10, Math.min(24, rowHeight * 0.6))

              return (
                <NoteChip
                  key={note.id}
                  note={note}
                  left={left}
                  top={top}
                  width={noteWidthPx}
                  height={noteHeight}
                  fontSize={fontSize}
                  isSelected={selectedId === note.id}
                  isOverlapping={isOverlapping}
                  onPointerDown={(event, mode) => startDrag(event, note, mode)}
                  onDoubleClick={(event) => {
                    event.stopPropagation()
                    onFocusLyric?.(note.id)
                  }}
                />
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
