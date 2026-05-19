import { useEffect, useRef } from 'react'

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.value = 880
    gain.gain.value = 0.2
    osc.start()
    osc.stop(ctx.currentTime + 0.4)
    setTimeout(() => ctx.close(), 500)
  } catch { /* audio not available */ }
}

const BEEP_COOLDOWN = 5000

export default function AlertBanner({ status }) {
  const duration   = status?.bad_posture_duration ?? 0
  const running    = status?.running ?? false
  const lastBeepThreshold = useRef(0)
  const lastBeepTime = useRef(0)

  useEffect(() => {
    if (running && duration >= 5) {
      const currentThreshold = Math.floor(duration / 5) * 5
      if (currentThreshold >= 5 && currentThreshold > lastBeepThreshold.current) {
        if (Date.now() - lastBeepTime.current > 1000) {
          beep()
          lastBeepTime.current = Date.now()
        }
        lastBeepThreshold.current = currentThreshold
      }
    } else {
      lastBeepThreshold.current = 0
    }
  }, [duration, running])

  if (!running || duration < 5) return null

  return (
    <div style={{
      position: 'fixed',
      top: 24,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 200,
      background: 'rgba(255, 250, 255, 0.82)',
      backdropFilter: 'blur(20px) saturate(140%)',
      WebkitBackdropFilter: 'blur(20px) saturate(140%)',
      border: '1px solid rgba(200, 40, 40, 0.22)',
      borderRadius: 100,
      padding: '12px 36px',
      textAlign: 'center',
      animation: 'pulse-alert 1.5s infinite',
      pointerEvents: 'none',
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.9)',
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: '50%',
        background: 'rgba(200, 40, 40, 0.85)',
        boxShadow: '0 0 8px rgba(200,40,40,0.5)',
        flexShrink: 0,
        animation: 'pulse-dot 1.5s infinite',
      }} />
      <div>
        <span style={{
          fontSize: '0.85rem', fontWeight: 800,
          color: 'rgba(170, 25, 25, 0.9)',
          letterSpacing: '0.08em', textTransform: 'uppercase',
        }}>
          Correct Your Posture
        </span>
        <span style={{
          fontSize: '0.8rem', fontWeight: 500,
          color: 'rgba(150, 30, 30, 0.6)',
          marginLeft: 10,
        }}>
          {duration}s
        </span>
      </div>
      <style>{`
        @keyframes pulse-alert {
          0%, 100% { box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 4px 20px rgba(200,40,40,0.1); }
          50%       { box-shadow: inset 0 1px 0 rgba(255,255,255,0.9), 0 4px 32px rgba(200,40,40,0.28); }
        }
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }
      `}</style>
    </div>
  )
}
