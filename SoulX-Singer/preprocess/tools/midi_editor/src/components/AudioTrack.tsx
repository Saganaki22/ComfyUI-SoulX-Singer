import { useEffect, useRef, forwardRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { PITCH_WIDTH } from '../constants'

export type AudioTrackProps = {
  audioUrl: string | null
  muted: boolean
  onSeek: (seconds: number) => void
  mediaElement?: HTMLAudioElement | null
  playheadSeconds: number
  gridSecondWidth: number
  minContentWidth?: number  // Minimum width to match MIDI editor area
}

export const AudioTrack = forwardRef<HTMLDivElement, AudioTrackProps>(
  ({ audioUrl, muted, onSeek, playheadSeconds, gridSecondWidth, minContentWidth = 0 }, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null)
    const waveRef = useRef<WaveSurfer | null>(null)
    const [waveWidth, setWaveWidth] = useState(0)

    useEffect(() => {
      if (!containerRef.current) return
      if (!audioUrl) {
        try {
          waveRef.current?.destroy()
        } catch {
          // ignore teardown errors
        }
        waveRef.current = null
        setWaveWidth(0)
        return
      }

      let cancelled = false

      // Clean up existing instance
      if (waveRef.current) {
        try {
          waveRef.current.destroy()
        } catch {
          // ignore teardown errors
        }
      }

      waveRef.current = WaveSurfer.create({
        container: containerRef.current,
        waveColor: '#4b64bc',
        progressColor: '#4b64bc',
        cursorColor: 'transparent',
        barWidth: 2,
        barGap: 2,
        height: 60,
        normalize: true,
        minPxPerSec: gridSecondWidth,
        interact: false,
        hideScrollbar: true,
        autoScroll: false,
      })

      waveRef.current.load(audioUrl).catch(() => null)
      waveRef.current.on('error', () => null)

      waveRef.current.on('ready', () => {
        if (cancelled || !waveRef.current) return
        const duration = waveRef.current.getDuration()
        const requiredWidth = duration * gridSecondWidth
        setWaveWidth(requiredWidth)
      })

      return () => {
        cancelled = true
        try {
          waveRef.current?.destroy()
        } catch {
          // ignore teardown errors
        }
        waveRef.current = null
      }
    }, [audioUrl, gridSecondWidth])

    useEffect(() => {
      if (!waveRef.current) return
      waveRef.current.setOptions({
        waveColor: muted ? '#9aa6b2' : '#4b64bc',
        progressColor: muted ? '#c0c9d4' : '#4b64bc',
      })
    }, [muted])

    if (!audioUrl) return null

    // Content width should be at least as wide as MIDI editor
    const contentWidth = Math.max(waveWidth, minContentWidth)

    return (
      <div
        className="audio-track-row"
        style={{ 
          display: 'flex', 
          borderBottom: '1px solid var(--border-soft)',
          height: '70px',
          flexShrink: 0
        }}
      >
        <div
          className="audio-gutter"
          style={{
            width: PITCH_WIDTH,
            flexShrink: 0,
            background: 'var(--panel-strong)',
            borderRight: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '11px',
            color: 'var(--text-muted)',
            fontWeight: 600,
          }}
        >
          AUDIO
        </div>

        {/* Scroll Mask - Controlled by parent via ref */}
        <div
          ref={ref}
          className="audio-scroll-mask"
          style={{
            flex: 1,
            overflow: 'hidden',
            position: 'relative',
            background: 'var(--panel-soft)',
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const scrollMask = e.currentTarget as HTMLDivElement
            const x = e.clientX - rect.left + scrollMask.scrollLeft
            const seconds = x / gridSecondWidth
            onSeek(seconds)
          }}
        >
          {/* Container that matches MIDI editor width */}
          <div 
            className="audio-content"
            style={{ 
              width: contentWidth > 0 ? contentWidth : '100%',
              height: '100%',
              position: 'relative'
            }}
          >
            {/* WaveSurfer container - only as wide as audio */}
            <div 
              ref={containerRef} 
              className="wave-container" 
              style={{ 
                width: waveWidth > 0 ? waveWidth : '100%',
                height: '100%',
                position: 'absolute',
                left: 0,
                top: 0
              }} 
            />

            {/* Custom Playhead */}
            <div
              className="audio-playhead"
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                width: '2px',
                background: '#ff7043',
                boxShadow: '0 0 12px rgba(255, 112, 67, 0.6)',
                left: playheadSeconds * gridSecondWidth,
                zIndex: 10,
                pointerEvents: 'none',
              }}
            />
          </div>
        </div>
      </div>
    )
  }
)
