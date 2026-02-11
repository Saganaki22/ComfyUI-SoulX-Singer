import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { PianoRoll } from './components/PianoRoll'
import { LyricTable } from './components/LyricTable'
import { AudioTrack } from './components/AudioTrack'
import { useMidiStore } from './store/useMidiStore'
import { exportMidi, importMidiFile } from './lib/midi'
import type { TimeSignature } from './types'
import { BASE_GRID_SECOND_WIDTH, BASE_ROW_HEIGHT, LOW_NOTE, HIGH_NOTE } from './constants'
import './App.css'

type PlayEvent = {
  time: number
  midi: number
  duration: number
  velocity: number
}

function App() {
  const {
    notes,
    tempo,
    timeSignature,
    selectedId,
    playhead,
    ppq,
    addNote,
    updateNote,
    removeNote,
    setNotes,
    setTempo,
    setTimeSignature,
    setPpq,
    select,
    setPlayhead,
  } = useMidiStore()

  const [status, setStatus] = useState('准备就绪')
  const [isPlaying, setIsPlaying] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>('light')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [audioDuration, setAudioDuration] = useState(0)
  const [midiVolume, setMidiVolume] = useState(80) // 0-100
  const [audioVolume, setAudioVolume] = useState(80) // 0-100
  const [horizontalZoom, setHorizontalZoom] = useState(1)
  const [verticalZoom, setVerticalZoom] = useState(1)
  const [focusLyricId, setFocusLyricId] = useState<string | null>(null)
  // Selection range for loop playback (in seconds)
  const [selectionStart, setSelectionStart] = useState<number | null>(null)
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null)
  const [isSelectingRange, setIsSelectingRange] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const audioInputRef = useRef<HTMLInputElement | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const partRef = useRef<Tone.Part<PlayEvent> | null>(null)
  const synthRef = useRef<Tone.PolySynth | null>(null)
  const rafRef = useRef<number | null>(null)
  const audioScrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    return () => {
      stopPlayback()
      synthRef.current?.dispose()
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  // Sync audio volume - also trigger when audioUrl changes (new audio loaded)
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = audioVolume / 100
    }
  }, [audioVolume, audioUrl])

  // Sync MIDI synth volume
  useEffect(() => {
    if (synthRef.current) {
      // Convert 0-100 to dB scale (-60 to 0)
      const dbValue = midiVolume === 0 ? -Infinity : (midiVolume / 100) * 60 - 60
      synthRef.current.volume.value = dbValue
    }
  }, [midiVolume])

  useEffect(() => {
    if (!audioUrl) return
    return () => {
      URL.revokeObjectURL(audioUrl)
    }
  }, [audioUrl])

  const ensureSynth = async () => {
    await Tone.start()
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination()
      // Apply current volume
      const dbValue = midiVolume === 0 ? -Infinity : (midiVolume / 100) * 60 - 60
      synthRef.current.volume.value = dbValue
    }
  }

  const playPreviewNote = useCallback(async (midi: number) => {
    await ensureSynth()
    const frequency = Tone.Frequency(midi, 'midi').toFrequency()
    synthRef.current?.triggerAttackRelease(frequency, '8n', Tone.now(), 0.7)
  }, [midiVolume])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!selectedId) return
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA'].includes(target.tagName)) return
      
      // Delete note
      if (event.key === 'Backspace' || event.key === 'Delete') {
        event.preventDefault()
        removeNote(selectedId)
        select(null)
        return
      }
      
      // Cmd/Ctrl + Up/Down to adjust pitch
      const isCmdOrCtrl = event.metaKey || event.ctrlKey
      if (isCmdOrCtrl && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        event.preventDefault()
        const selectedNote = notes.find(n => n.id === selectedId)
        if (!selectedNote) return
        
        const delta = event.key === 'ArrowUp' ? 1 : -1
        const newMidi = Math.max(LOW_NOTE, Math.min(HIGH_NOTE, selectedNote.midi + delta))
        
        if (newMidi !== selectedNote.midi) {
          updateNote(selectedId, { midi: newMidi })
          playPreviewNote(newMidi)
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedId, notes, removeNote, select, updateNote, playPreviewNote])

  const noteEvents = useMemo<PlayEvent[]>(
    () =>
      notes.map((note) => ({
        time: (60 / tempo) * note.start,
        duration: (60 / tempo) * note.duration,
        midi: note.midi,
        velocity: note.velocity,
      })),
    [notes, tempo],
  )

  const beatToSeconds = (beat: number) => beat * (60 / tempo)
  const secondsToBeat = (seconds: number) => seconds / (60 / tempo)
  const seekBySeconds = (deltaSeconds: number) => {
    const maxNoteEnd = notes.reduce((acc, n) => Math.max(acc, n.start + n.duration), 0)
    const maxBeat = Math.max(secondsToBeat(audioDuration), maxNoteEnd)
    const nextSeconds = Math.max(0, Math.min(beatToSeconds(maxBeat), beatToSeconds(playhead) + deltaSeconds))
    seekToBeat(secondsToBeat(nextSeconds))
  }

  const gridSecondWidth = BASE_GRID_SECOND_WIDTH * horizontalZoom
  const rowHeight = BASE_ROW_HEIGHT * verticalZoom

  // Calculate MIDI content width to sync with audio track
  const midiContentWidth = useMemo(() => {
    const noteEndSeconds = notes.reduce((acc, n) => {
      const endBeat = n.start + n.duration
      return Math.max(acc, beatToSeconds(endBeat))
    }, 8)
    const maxSeconds = Math.max(noteEndSeconds + 10, audioDuration + 10, 30)
    return maxSeconds * gridSecondWidth
  }, [notes, audioDuration, gridSecondWidth, beatToSeconds])

  const seekToBeat = (beat: number) => {
    setPlayhead(beat)
    Tone.Transport.seconds = beatToSeconds(beat)
    if (audioRef.current) {
      audioRef.current.currentTime = beatToSeconds(beat)
    }
  }

  const schedulePlayback = async () => {
    if (!notes.length && !audioUrl) return
    await ensureSynth()
    partRef.current?.dispose()
    Tone.Transport.cancel()
    Tone.Transport.stop()
    Tone.Transport.bpm.value = tempo

    // Determine playback range
    const hasSelection = selectionStart !== null && selectionEnd !== null && selectionEnd > selectionStart
    const startSeconds = hasSelection ? selectionStart : beatToSeconds(playhead)
    const endSeconds = hasSelection ? selectionEnd : null

    Tone.Transport.seconds = startSeconds

    // Filter notes within selection range if applicable
    const filteredEvents = hasSelection
      ? noteEvents.filter(e => e.time >= startSeconds && e.time < endSeconds!)
      : noteEvents

    if (filteredEvents.length) {
      partRef.current = new Tone.Part((time, event) => {
        if (midiVolume === 0) return
        const frequency = Tone.Frequency(event.midi, 'midi').toFrequency()
        synthRef.current?.triggerAttackRelease(frequency, event.duration, time, event.velocity)
      }, filteredEvents)
      partRef.current.start(0)
    }
    Tone.Transport.start()
    if (audioRef.current && audioUrl) {
      audioRef.current.currentTime = startSeconds
      if (audioVolume > 0) {
        audioRef.current.play().catch(() => null)
      }
    }
    setIsPlaying(true)
    setStatus(hasSelection ? '选区回放中...' : '正在回放...')

    const tick = () => {
      const seconds =
        audioRef.current && audioUrl && !audioRef.current.paused
          ? audioRef.current.currentTime
          : Tone.Transport.seconds
      
      // Stop at selection end
      if (endSeconds !== null && seconds >= endSeconds) {
        pausePlayback()
        seekToBeat(secondsToBeat(selectionStart!))
        setStatus('选区播放完毕')
        return
      }
      
      const beat = seconds / (60 / tempo)
      setPlayhead(beat)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  const stopPlayback = () => {
    Tone.Transport.stop()
    Tone.Transport.cancel()
    partRef.current?.dispose()
    partRef.current = null
    setIsPlaying(false)
    setPlayhead(0)
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  const pausePlayback = () => {
    Tone.Transport.stop()
    partRef.current?.dispose()
    partRef.current = null
    setIsPlaying(false)
    if (audioRef.current) {
      audioRef.current.pause()
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  const handlePlayToggle = async () => {
    if (isPlaying) {
      pausePlayback()
      setStatus('已暂停')
    } else {
      await schedulePlayback()
    }
  }

  const handleImportClick = () => fileInputRef.current?.click()
  const handleAudioImportClick = () => audioInputRef.current?.click()

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const snapshot = await importMidiFile(file)
      setNotes(snapshot.notes)
      setTempo(snapshot.tempo)
      setTimeSignature(snapshot.timeSignature as TimeSignature)
      setPpq(snapshot.ppq)  // Preserve original ppq for accurate export
      setStatus(`已载入 ${file.name}`)
    } catch (error) {
      console.error(error)
      setStatus('导入失败，请确认文件合法')
    } finally {
      event.target.value = ''
    }
  }

  const handleAudioChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    // Validate audio file type
    const validAudioTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/flac', 'audio/mp4', 'audio/aac', 'audio/x-m4a']
    const validExtensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac']
    const fileName = file.name.toLowerCase()
    const isValidType = validAudioTypes.includes(file.type) || file.type.startsWith('audio/')
    const isValidExtension = validExtensions.some(ext => fileName.endsWith(ext))
    
    if (!isValidType && !isValidExtension) {
      setStatus(`不支持的文件格式，请选择音频文件（${validExtensions.join(', ')}）`)
      event.target.value = ''
      return
    }
    
    const url = URL.createObjectURL(file)
    setAudioUrl(url)
    setStatus(`已载入音频 ${file.name}`)
    event.target.value = ''
  }

  // Check for overlapping notes (any pitch)
  const getOverlappingNotes = () => {
    const overlapping: string[] = []
    const sortedNotes = [...notes].sort((a, b) => a.start - b.start)
    const EPSILON = 0.05 // Tolerance for floating point comparison
    
    for (let i = 0; i < sortedNotes.length; i++) {
      for (let j = i + 1; j < sortedNotes.length; j++) {
        const noteA = sortedNotes[i]
        const noteB = sortedNotes[j]
        const noteAEnd = noteA.start + noteA.duration
        // If noteB starts at or after noteA ends (with tolerance), no overlap
        if (noteB.start >= noteAEnd - EPSILON) break
        // True overlap: noteB starts before noteA ends
        if (!overlapping.includes(noteA.id)) overlapping.push(noteA.id)
        if (!overlapping.includes(noteB.id)) overlapping.push(noteB.id)
      }
    }
    return overlapping
  }

  // Auto-fix overlapping notes by trimming the first note to end where the second begins
  const handleFixOverlaps = () => {
    const sortedNotes = [...notes].sort((a, b) => a.start - b.start)
    let fixCount = 0
    
    for (let i = 0; i < sortedNotes.length - 1; i++) {
      const noteA = sortedNotes[i]
      const noteB = sortedNotes[i + 1]
      const noteAEnd = noteA.start + noteA.duration
      
      // If noteA overlaps with noteB
      if (noteAEnd > noteB.start) {
        // Trim noteA to end at noteB's start
        const newDuration = Math.max(0.01, noteB.start - noteA.start)
        updateNote(noteA.id, { duration: newDuration })
        fixCount++
      }
    }
    
    if (fixCount > 0) {
      setStatus(`已修复 ${fixCount} 个重叠音符`)
    } else {
      setStatus('没有检测到重叠音符')
    }
  }

  const handleExport = () => {
    const overlapping = getOverlappingNotes()
    if (overlapping.length > 0) {
      const confirm = window.confirm(
        `检测到 ${overlapping.length} 个音符存在时间重叠（标红色的音符），这可能导致播放异常。\n\n是否仍要导出？`
      )
      if (!confirm) return
    }
    
    const blob = exportMidi({ notes, tempo, timeSignature, ppq })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'vocal-midi.mid'
    anchor.click()
    URL.revokeObjectURL(url)
    setStatus('已导出包含歌词的 MIDI 文件')
  }


  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">歌声 MIDI 编辑器</p>
          <h1>Lyric-ready Piano Roll</h1>
          <p className="muted">导入、拖拽、实时修改歌词并导出标准 MIDI。</p>
        </div>
        <div className="actions">
          <button className="icon-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? (
              <span className="icon" aria-label="切换到亮色">
                ☀️
              </span>
            ) : (
              <span className="icon" aria-label="切换到暗色">
                🌙
              </span>
            )}
          </button>
          <button className="primary" onClick={handleImportClick}>
            导入 MIDI
          </button>
          <button className="primary" onClick={handleExport}>
            导出含歌词 MIDI
          </button>
          <button className="soft" onClick={handleFixOverlaps} title="自动消除重叠：将重叠音符的音尾提前到下一个音的音头">
            消除重叠
          </button>
          <input ref={fileInputRef} type="file" accept=".mid,.midi" className="sr-only" onChange={handleFileChange} />
        </div>
      </header>

      <section className="audio-bar">
        <div className="audio-left">
          <button className="ghost" onClick={handleAudioImportClick}>
            对齐音频导入
          </button>
          <input
            ref={audioInputRef}
            type="file"
            accept=".mp3,.wav,.ogg,.flac,.m4a,.aac"
            className="sr-only"
            onChange={handleAudioChange}
          />
          <span className="audio-hint">导入后显示音频波形并与 MIDI 同步走带</span>
        </div>
        <div className="audio-right">
          <div className="volume-control">
            <span className="volume-label">MIDI</span>
            <input
              type="range"
              min={0}
              max={100}
              value={midiVolume}
              onChange={(e) => setMidiVolume(Number(e.target.value))}
              className="volume-slider"
            />
            <span className="volume-value">{midiVolume}%</span>
          </div>
          <div className="volume-control">
            <span className="volume-label">音频</span>
            <input
              type="range"
              min={0}
              max={100}
              value={audioVolume}
              onChange={(e) => setAudioVolume(Number(e.target.value))}
              className="volume-slider"
            />
            <span className="volume-value">{audioVolume}%</span>
          </div>
        </div>
      </section>

      <section className="panel panel-split">
        <div className="panel-main">
          {audioUrl && (
            <AudioTrack
              key={audioUrl}
              ref={audioScrollRef}
              audioUrl={audioUrl}
              muted={audioVolume === 0}
              onSeek={(seconds) => seekToBeat(secondsToBeat(seconds))}
              playheadSeconds={beatToSeconds(playhead)}
              gridSecondWidth={gridSecondWidth}
              minContentWidth={midiContentWidth}
            />
          )}
          <PianoRoll
            notes={notes}
            selectedId={selectedId}
            timeSignature={timeSignature}
            tempo={tempo}
            playhead={playhead}
            selectionStart={selectionStart}
            selectionEnd={selectionEnd}
            onAddNote={addNote}
            onSelect={select}
            onUpdateNote={updateNote}
            onSeek={seekToBeat}
            onScroll={(left) => {
              if (audioScrollRef.current) {
                audioScrollRef.current.scrollLeft = left
              }
            }}
            onZoom={(deltaH, deltaV) => {
              if (deltaH !== 0) {
                setHorizontalZoom(prev => Math.max(0.5, prev + deltaH))
              }
              if (deltaV !== 0) {
                setVerticalZoom(prev => Math.max(0.6, Math.min(2.5, prev + deltaV)))
              }
            }}
            onPlayNote={playPreviewNote}
            onFocusLyric={(noteId) => {
              select(noteId)
              setFocusLyricId(noteId)
            }}
            onSelectionChange={(start, end) => {
              setSelectionStart(start)
              setSelectionEnd(end)
            }}
            isSelectingRange={isSelectingRange}
            audioDuration={audioDuration}
            gridSecondWidth={gridSecondWidth}
            rowHeight={rowHeight}
          />
        </div>
        <aside className="panel-side">
          <div className="controls">
            <div className="toggle" style={{ justifyContent: 'space-between' }}>
              <span>水平缩放</span>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.1}
                value={Math.min(horizontalZoom, 10)}
                onChange={(e) => setHorizontalZoom(Number(e.target.value))}
                style={{ width: '140px' }}
              />
              <span style={{ width: 42, textAlign: 'right' }}>{horizontalZoom.toFixed(1)}x</span>
            </div>
            <div className="toggle" style={{ justifyContent: 'space-between' }}>
              <span>垂直缩放</span>
              <input
                type="range"
                min={0.6}
                max={2.5}
                step={0.1}
                value={verticalZoom}
                onChange={(e) => setVerticalZoom(Number(e.target.value))}
                style={{ width: '140px' }}
              />
              <span style={{ width: 42, textAlign: 'right' }}>{verticalZoom.toFixed(1)}x</span>
            </div>
            <div className="transport">
              <button 
                className="soft" 
                onClick={() => {
                   setPlayhead(0)
                   seekToBeat(0)
                }}
                title="回到开头"
              >
                ⏮
              </button>
              <button
                className="soft"
                onClick={() => seekBySeconds(-2)}
                title="后退 2 秒"
              >
                ⏪ 2s
              </button>
              <button 
                 className="primary" 
                 onClick={handlePlayToggle} 
                 disabled={!notes.length && !audioUrl}
                 title={isPlaying ? "暂停" : (selectionStart !== null && selectionEnd !== null ? "播放选区" : "播放")}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <button
                className="soft"
                onClick={() => seekBySeconds(2)}
                title="前进 2 秒"
              >
                2s ⏩
              </button>
              <button 
                 className="soft"
                 onClick={() => {
                    // Logic to find end of song? Max note end or audio duration
                    const maxNoteEnd = notes.reduce((acc, n) => Math.max(acc, n.start + n.duration), 0)
                    seekToBeat(Math.max(secondsToBeat(audioDuration), maxNoteEnd))
                 }}
                 title="回到结尾"
              >
                ⏭
              </button>
            </div>
            <div className="selection-controls">
              <button
                className={`soft selection-btn ${isSelectingRange ? 'active' : ''}`}
                onClick={() => setIsSelectingRange(!isSelectingRange)}
                title={isSelectingRange ? "退出选区模式" : "设置选区：在时间轴上拖拽选择播放范围"}
              >
                {isSelectingRange ? '📍 选区中' : '📍 设选区'}
              </button>
              {selectionStart !== null && selectionEnd !== null && (
                <>
                  <span className="selection-info">
                    {selectionStart.toFixed(1)}s - {selectionEnd.toFixed(1)}s
                  </span>
                  <button
                    className="soft"
                    onClick={() => {
                      setSelectionStart(null)
                      setSelectionEnd(null)
                    }}
                    title="清除选区"
                  >
                    ✕
                  </button>
                </>
              )}
            </div>
            <div className="status">{status}</div>
          </div>
          <div className="lyric-container">
            <LyricTable 
              notes={notes} 
              selectedId={selectedId} 
              tempo={tempo} 
              focusLyricId={focusLyricId}
              onSelect={select} 
              onUpdate={updateNote}
              onFocusHandled={() => setFocusLyricId(null)}
            />
          </div>
        </aside>
      </section>
      <audio 
        ref={audioRef} 
        src={audioUrl ?? undefined} 
        preload="auto" 
        className="sr-only" 
        onLoadedMetadata={(e) => {
          setAudioDuration(e.currentTarget.duration)
          // Ensure volume is set when audio loads
          e.currentTarget.volume = audioVolume / 100
        }}
      />
    </div>
  )
}

export default App
