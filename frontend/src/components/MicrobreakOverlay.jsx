export default function MicrobreakOverlay({ status, onEndBreak }) {
  if (!status?.in_microbreak) return null

  const elapsed   = status.microbreak_elapsed ?? 0
  const countdown = status.microbreak_countdown ?? 60
  const exercise  = status.microbreak_exercise ?? ''
  const progress  = Math.min(1, elapsed / 60)

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 300,
      background: 'rgba(250, 248, 255, 0.75)',
      backdropFilter: 'blur(30px) saturate(150%)',
      WebkitBackdropFilter: 'blur(30px) saturate(150%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      <div style={{
        background: 'rgba(255, 255, 255, 0.72)',
        backdropFilter: 'blur(20px) saturate(140%)',
        WebkitBackdropFilter: 'blur(20px) saturate(140%)',
        border: '1px solid rgba(130, 60, 210, 0.14)',
        borderRadius: 28,
        padding: '48px 56px',
        maxWidth: 520,
        width: '90%',
        textAlign: 'center',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.9), 0 20px 60px rgba(100,45,175,0.12)',
      }}>

        {/* Section tag */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          background: 'rgba(110,50,200,0.07)',
          border: '1px solid rgba(110,50,200,0.15)',
          borderRadius: 100,
          padding: '6px 18px',
          marginBottom: 20,
          fontSize: '0.72rem', fontWeight: 800,
          letterSpacing: '0.28em', textTransform: 'uppercase',
          color: 'rgba(100,45,185,0.85)',
        }}>
          <div style={{
            width: 5, height: 5, borderRadius: '50%',
            background: 'linear-gradient(135deg, rgba(110,40,190,0.8), rgba(0,185,220,0.7))',
          }} />
          Microbreak
        </div>

        {/* Countdown */}
        <div style={{
          fontFamily: "'Playfair Display', serif",
          fontStyle: 'italic',
          fontSize: 'clamp(2.5rem, 6vw, 3.5rem)',
          fontWeight: 400,
          background: 'linear-gradient(125deg, rgba(75,15,165,0.85), rgba(155,85,230,0.8), rgba(0,190,225,0.72))',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          lineHeight: 1,
          marginBottom: 8,
        }}>
          {countdown}s
        </div>
        <div style={{
          fontSize: '0.82rem', fontWeight: 600,
          color: 'rgba(80,35,145,0.45)',
          letterSpacing: '0.05em',
          marginBottom: 28,
        }}>
          {elapsed}s elapsed · {countdown}s remaining
        </div>

        {/* Progress bar */}
        <div style={{
          background: 'rgba(110,50,200,0.07)',
          border: '1px solid rgba(110,50,200,0.1)',
          borderRadius: 100,
          height: 6,
          marginBottom: 32,
          overflow: 'hidden',
        }}>
          <div style={{
            background: 'linear-gradient(90deg, rgba(110,40,190,0.7), rgba(0,185,220,0.65))',
            height: '100%',
            width: `${progress * 100}%`,
            transition: 'width 1s linear',
            borderRadius: 100,
          }} />
        </div>

        {/* Exercise label */}
        <div style={{
          fontSize: '0.66rem', fontWeight: 800,
          letterSpacing: '0.24em', textTransform: 'uppercase',
          color: 'rgba(100,45,185,0.4)',
          marginBottom: 10,
        }}>
          Current Exercise
        </div>
        <div style={{
          fontSize: '0.95rem', fontWeight: 500,
          color: 'rgba(50, 20, 95, 0.72)',
          lineHeight: 1.7,
          minHeight: 56,
          marginBottom: 32,
        }}>
          {exercise}
        </div>

        {/* End break button */}
        <button
          onClick={onEndBreak}
          style={{
            background: 'rgba(120,55,200,0.07)',
            border: '1px solid rgba(120,55,200,0.18)',
            borderRadius: 100,
            color: 'rgba(85,30,165,0.75)',
            cursor: 'pointer',
            fontFamily: "'Nunito', sans-serif",
            fontSize: '0.8rem',
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            padding: '11px 28px',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.88)',
            transition: 'all 0.3s cubic-bezier(0.23, 1, 0.32, 1)',
          }}
          onMouseOver={e => {
            e.currentTarget.style.background = 'linear-gradient(135deg,rgba(85,20,170,0.82),rgba(100,55,200,0.75),rgba(0,170,210,0.7))'
            e.currentTarget.style.color = 'rgba(255,255,255,0.95)'
            e.currentTarget.style.borderColor = 'transparent'
            e.currentTarget.style.transform = 'translateY(-2px)'
          }}
          onMouseOut={e => {
            e.currentTarget.style.background = 'rgba(120,55,200,0.07)'
            e.currentTarget.style.color = 'rgba(85,30,165,0.75)'
            e.currentTarget.style.borderColor = 'rgba(120,55,200,0.18)'
            e.currentTarget.style.transform = 'translateY(0)'
          }}
        >
          End Break Early
        </button>
      </div>
    </div>
  )
}
