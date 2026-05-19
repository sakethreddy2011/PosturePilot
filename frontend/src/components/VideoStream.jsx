import { useRef, useState, useEffect } from 'react'
import SkeletonOverlay from './SkeletonOverlay.jsx'

export default function VideoStream({ status }) {
  const imgRef = useRef(null)
  const [videoDims, setVideoDims] = useState({ width: 0, height: 0 })

  const running = status?.running
  useEffect(() => {
    const el = imgRef.current
    if (!el) return
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setVideoDims({ width, height })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [running])

  const poseDetected = status?.pose_detected
  const landmarks    = status?.landmarks
  const showNodes    = status?.show_nodes ?? true

  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <div style={{ position: 'relative', display: 'inline-flex' }}>
        {running ? (
          <img
            ref={imgRef}
            src="/video_feed"
            alt="posture feed"
            style={{
              display: 'block',
              width: '640px',
              maxWidth: '100%',
              aspectRatio: '4/3',
              height: 'auto',
              margin: '0 auto',
              borderRadius: 20,
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.85), 0 8px 40px rgba(100,45,175,0.12), 0 0 0 1px rgba(130,60,210,0.1)',
            }}
          />
        ) : (
          <div
            ref={imgRef}
            className="glass-panel"
            style={{
              width: '640px',
              maxWidth: '100%',
              aspectRatio: '4/3',
              height: 'auto',
              margin: '0 auto',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 24,
              color: 'rgba(55, 20, 110, 0.65)',
              fontSize: '1.15rem',
              letterSpacing: '0.04em',
              backgroundImage: [
                'linear-gradient(rgba(100,45,175,0.03) 1px, transparent 1px)',
                'linear-gradient(90deg, rgba(0,175,215,0.025) 1px, transparent 1px)',
              ].join(', '),
              backgroundSize: '32px 32px',
            }}
          >
            <div style={{ fontSize: '3rem', marginBottom: 16, opacity: 0.35 }}>📷</div>
            <div style={{
              fontFamily: "'Playfair Display', serif",
              fontStyle: 'italic',
              fontSize: '1.3rem',
              fontWeight: 400,
              color: 'rgba(45, 15, 95, 0.75)',
              marginBottom: 8,
            }}>
              Camera Offline
            </div>
            <div style={{
              fontSize: '0.85rem',
              color: 'rgba(55, 20, 110, 0.7)',
              fontWeight: 600,
            }}>
              Press Start to begin analyzing
            </div>
          </div>
        )}

        {running && !poseDetected && videoDims.width > 50 && (
          <div style={{
            position: 'absolute',
            top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'rgba(255,255,255,0.82)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            border: '1px solid rgba(200,50,50,0.2)',
            borderRadius: 100,
            padding: '8px 20px',
            color: 'rgba(180, 30, 30, 0.85)',
            fontSize: '0.85rem',
            fontWeight: 700,
            letterSpacing: '0.06em',
            pointerEvents: 'none',
          }}>
            Position yourself in frame
          </div>
        )}

        <SkeletonOverlay
          landmarks={landmarks}
          videoDims={videoDims}
          show={running && showNodes && poseDetected}
        />
      </div>
    </div>
  )
}
