function Btn({ label, onClick, disabled, variant = 'default' }) {
  // Landing-theme colour palette: purple/cyan/light
  const colors = {
    default: {
      bg: 'rgba(120, 55, 200, 0.06)',
      border: 'rgba(120, 55, 200, 0.16)',
      text: 'rgba(55, 20, 110, 0.88)',
      hoverBg: 'linear-gradient(135deg, rgba(85,20,170,0.82), rgba(100,55,200,0.75), rgba(0,170,210,0.7))',
      glow: 'rgba(100,45,185,0.14)',
    },
    green: {
      bg: 'linear-gradient(135deg, rgba(85,20,170,0.82), rgba(100,55,200,0.75), rgba(0,170,210,0.7))',
      border: 'transparent',
      text: 'rgba(255,255,255,0.95)',
      hoverBg: 'linear-gradient(135deg, rgba(70,10,155,0.92), rgba(85,40,185,0.88), rgba(0,155,200,0.82))',
      glow: 'rgba(100,40,185,0.22)',
    },
    red: {
      bg: 'rgba(200, 50, 50, 0.08)',
      border: 'rgba(200, 50, 50, 0.22)',
      text: 'rgba(155, 20, 20, 0.95)',
      hoverBg: 'rgba(200, 50, 50, 0.18)',
      glow: 'rgba(200,50,50,0.12)',
    },
    blue: {
      bg: 'rgba(0, 185, 220, 0.07)',
      border: 'rgba(0, 185, 220, 0.2)',
      text: 'rgba(0, 110, 155, 0.95)',
      hoverBg: 'rgba(0, 185, 220, 0.15)',
      glow: 'rgba(0,185,220,0.12)',
    },
    amber: {
      bg: 'rgba(160, 90, 0, 0.07)',
      border: 'rgba(200, 130, 0, 0.22)',
      text: 'rgba(120, 65, 0, 0.92)',
      hoverBg: 'rgba(200, 130, 0, 0.15)',
      glow: 'rgba(200,130,0,0.1)',
    },
  }
  const c = colors[variant] ?? colors.default

  const base = {
    background: disabled ? 'rgba(200,190,220,0.08)' : c.bg,
    border: `1px solid ${disabled ? 'rgba(130,60,210,0.08)' : c.border}`,
    borderRadius: 100,
    color: disabled ? 'rgba(110,80,160,0.45)' : c.text,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: '0.82rem',
    fontWeight: 700,
    fontFamily: "'Nunito', sans-serif",
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    padding: '10px 22px',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    boxShadow: disabled ? 'none' : `inset 0 1px 0 rgba(255,255,255,0.85), 0 3px 12px ${c.glow}`,
    transition: 'all 0.3s cubic-bezier(0.23, 1, 0.32, 1)',
    whiteSpace: 'nowrap',
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={base}
      onMouseOver={e => {
        if (!disabled) {
          e.currentTarget.style.background = c.hoverBg
          e.currentTarget.style.color = variant === 'green' ? 'rgba(255,255,255,0.95)' : c.text
          e.currentTarget.style.transform = 'translateY(-2px) scale(1.03)'
          e.currentTarget.style.borderColor = 'transparent'
          e.currentTarget.style.boxShadow = `0 6px 20px ${c.glow}, inset 0 1px 0 rgba(255,255,255,0.2)`
        }
      }}
      onMouseOut={e => {
        if (!disabled) {
          e.currentTarget.style.background = c.bg
          e.currentTarget.style.color = c.text
          e.currentTarget.style.transform = 'translateY(0) scale(1)'
          e.currentTarget.style.borderColor = c.border
          e.currentTarget.style.boxShadow = `inset 0 1px 0 rgba(255,255,255,0.85), 0 3px 12px ${c.glow}`
        }
      }}
    >
      {label}
    </button>
  )
}

export default function ControlBar({ status, onStart, onStop, onCalibrate, onToggleNodes, onEndBreak }) {
  const running      = status?.running ?? false
  const poseDetected = status?.pose_detected ?? false
  const inBreak      = status?.in_microbreak ?? false
  const showNodes    = status?.show_nodes ?? true

  return (
    <div className="glass-panel" style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: 12,
      padding: '16px 24px',
      alignItems: 'center',
      borderRadius: 20,
    }}>
      {!running
        ? <Btn label="▶ Start"  onClick={onStart}  variant="green" />
        : <Btn label="■ Stop"   onClick={onStop}   variant="red"   />
      }
      <Btn label="⊕ Calibrate"             onClick={onCalibrate}     disabled={!running || !poseDetected} variant="blue"    />
      <Btn label={showNodes ? 'Nodes: ON' : 'Nodes: OFF'} onClick={onToggleNodes} disabled={!running}  variant="default"  />
    </div>
  )
}
