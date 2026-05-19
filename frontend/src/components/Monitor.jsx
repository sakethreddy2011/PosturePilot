import { useState } from 'react'
import { useStatus } from '../hooks/useStatus.js'
import { startCamera, stopCamera, calibrate, toggleNodes, endMicrobreak } from '../api/client.js'
import VideoStream from './VideoStream.jsx'
import Dashboard from './Dashboard.jsx'
import ControlBar from './ControlBar.jsx'
import AlertBanner from './AlertBanner.jsx'
import MicrobreakOverlay from './MicrobreakOverlay.jsx'
import ReportModal from './ReportModal.jsx'
import { Link } from 'react-router-dom'

export default function Monitor() {
  const status = useStatus(500)
  const [report, setReport] = useState(null)

  const handleStart     = async () => { await startCamera() }
  const handleStop      = async () => {
    const res = await stopCamera()
    if (res?.ok && res?.report) {
      setReport({ 
        text: res.report, 
        screenshots: res.screenshots || [],
        pdf_report: res.pdf_report || null
      })
    }
  }
  const handleCalibrate = async () => { const res = await calibrate(); if (!res.ok) alert(res.message) }
  const handleToggleNodes = async () => { await toggleNodes() }
  const handleEndBreak  = async () => { await endMicrobreak() }

  return (
    <>
      {/* App shell */}
      <div style={{
        position: 'relative',
        zIndex: 1,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '32px 48px',
        gap: 24,
        maxWidth: 1440,
        margin: '0 auto',
      }}>

        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 style={{
              fontFamily: "'Playfair Display', serif",
              fontStyle: 'italic',
              fontSize: 'clamp(1.6rem, 3vw, 2.4rem)',
              fontWeight: 400,
              background: 'linear-gradient(125deg, rgba(75,15,165,0.9), rgba(155,85,230,0.82), rgba(0,190,225,0.75))',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              letterSpacing: '-0.01em',
              marginBottom: 6,
            }}>
              PosturePilot
            </h1>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              fontSize: '0.82rem', color: 'rgba(55, 20, 110, 0.82)',
              fontWeight: 700, letterSpacing: '0.06em',
            }}>
              {status ? (
                <>
                  <div style={{
                    width: 7, height: 7, borderRadius: '50%',
                    background: status.running
                      ? 'linear-gradient(135deg,rgba(110,40,190,0.9),rgba(0,185,220,0.8))'
                      : 'rgba(180,160,210,0.4)',
                    boxShadow: status.running ? '0 0 8px rgba(110,40,190,0.4)' : 'none',
                    flexShrink: 0,
                  }} />
                  {status.running ? 'Session Active' : 'Session Stopped'}
                </>
              ) : 'Connecting...'}
            </div>
          </div>

          {/* Back to landing */}
          <Link to="/" style={{
            fontFamily: "'Nunito', sans-serif",
            fontSize: '0.78rem',
            fontWeight: 700,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            textDecoration: 'none',
            color: 'rgba(75, 25, 155, 0.82)',
            padding: '8px 18px',
            borderRadius: 100,
            border: '1px solid rgba(120,55,200,0.22)',
            background: 'rgba(120,55,200,0.07)',
            transition: 'all 0.25s',
          }}
          onMouseOver={e => { e.currentTarget.style.color='rgba(75,20,155,1)'; e.currentTarget.style.background='rgba(120,55,200,0.13)' }}
          onMouseOut={e => { e.currentTarget.style.color='rgba(75,25,155,0.82)'; e.currentTarget.style.background='rgba(120,55,200,0.07)' }}
          >
            ← Back
          </Link>
        </div>

        {/* Control bar */}
        <ControlBar
          status={status}
          onStart={handleStart}
          onStop={handleStop}
          onCalibrate={handleCalibrate}
          onToggleNodes={handleToggleNodes}
          onEndBreak={handleEndBreak}
        />

        {/* Main layout: video + dashboard */}
        <div style={{ display: 'flex', gap: 28, flex: 1, alignItems: 'flex-start' }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <VideoStream status={status} />
          </div>
          <div style={{ flexShrink: 0, width: 320 }}>
            <Dashboard status={status} />
          </div>
        </div>

        {/* Overlays */}
        <AlertBanner status={status} />
        <MicrobreakOverlay status={status} onEndBreak={handleEndBreak} />
        <ReportModal report={report} onClose={() => setReport(null)} />
      </div>
    </>
  )
}
