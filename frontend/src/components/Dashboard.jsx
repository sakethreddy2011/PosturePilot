const STATE_COLORS = {
  GOOD: {
    bg:     'rgba(110, 40, 190, 0.07)',
    border: 'rgba(110, 40, 190, 0.22)',
    text:   'rgba(85, 20, 170, 0.9)',
    glow:   '0 0 20px rgba(110, 40, 190, 0.14)',
  },
  BAD: {
    bg:     'rgba(210, 40, 40, 0.07)',
    border: 'rgba(210, 40, 40, 0.22)',
    text:   'rgba(180, 25, 25, 0.9)',
    glow:   '0 0 20px rgba(210, 40, 40, 0.12)',
  },
}

const DEFECT_LABELS = {
  hunching:     'Hunching',
  lateral_lean: 'Lat Lean',
  hip_sliding:  'Hip Slide',
}

function fmtSecs(s) {
  if (s < 60) return `${s}s`
  return `${Math.floor(s / 60)}m ${s % 60}s`
}

export default function Dashboard({ status }) {
  if (!status) return null

  const state       = status.state ?? 'GOOD'
  const colors      = STATE_COLORS[state] ?? STATE_COLORS.GOOD
  const score       = status.score
  const defects     = status.defects ?? []
  const defectTimes = status.defect_times ?? {}
  const nextBreakIn = status.next_break_in ?? 0
  const calibrated  = status.calibrated
  const running     = status.running ?? false

  const dominant = defects.length > 0
    ? defects.reduce((a, b) => b.severity > a.severity ? b : a)
    : null

  return (
    <div className="glass-panel" style={{
      borderRadius: 24,
      padding: '28px 24px',
      minWidth: 280,
      display: 'flex',
      flexDirection: 'column',
      gap: 20,
    }}>

      {/* Section label */}
      <div style={{
        fontSize: '0.68rem',
        fontWeight: 800,
        letterSpacing: '0.28em',
        textTransform: 'uppercase',
        color: 'rgba(75, 30, 155, 0.82)',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
      }}>
        <div style={{
          width: 5, height: 5, borderRadius: '50%',
          background: 'linear-gradient(135deg, rgba(110,40,190,0.8), rgba(0,185,220,0.7))',
          boxShadow: '0 0 6px rgba(110,40,190,0.3)',
        }} />
        Dashboard
      </div>

      {/* Everything below is ONLY shown when session is running */}
      {running ? (
        <>
          {/* State chip — only when calibrated */}
          {calibrated && (
            <div style={{
              background: colors.bg,
              border: `1px solid ${colors.border}`,
              borderRadius: 16,
              padding: '14px 16px',
              textAlign: 'center',
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.7), ${colors.glow}`,
              transition: 'all 0.35s ease',
            }}>
              <div style={{
                fontFamily: "'Playfair Display', serif",
                fontStyle: 'italic',
                fontSize: '1.5rem',
                fontWeight: 400,
                color: colors.text,
                letterSpacing: '0.03em',
              }}>
                {state}
              </div>
              {score !== null && score !== undefined && (
                <div style={{
                  fontSize: '0.82rem',
                  color: 'rgba(55, 20, 110, 0.8)',
                  marginTop: 5,
                  fontWeight: 700,
                }}>
                  Score: <span style={{ color: 'rgba(80,35,145,0.85)' }}>{score.toFixed(1)}</span>
                </div>
              )}
            </div>
          )}

          {/* Active defect warning */}
          {dominant && (
            <div style={{
              background: 'rgba(210, 100, 0, 0.06)',
              border: '1px solid rgba(210, 100, 0, 0.18)',
              borderRadius: 12,
              padding: '10px 14px',
              fontSize: '0.83rem',
              color: 'rgba(170, 70, 0, 0.88)',
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <span style={{ fontSize: '1.1rem' }}>⚠</span>
              {dominant.key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
            </div>
          )}

          {/* Per-defect timers — only when calibrated */}
          {calibrated && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{
                fontSize: '0.65rem', fontWeight: 800,
                letterSpacing: '0.2em', textTransform: 'uppercase',
                color: 'rgba(75, 30, 155, 0.78)',
              }}>
                Duration Tracking
              </div>
              {Object.entries(DEFECT_LABELS).map(([key, label]) => {
                const secs     = defectTimes[key] ?? 0
                const isActive = defects.some(d => d.key === key)
                return (
                  <div key={key} style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', fontSize: '0.85rem',
                  }}>
                    <span style={{
                      color: isActive ? 'rgba(190,30,30,0.9)' : 'rgba(55, 20, 110, 0.78)',
                      fontWeight: isActive ? 700 : 600,
                      display: 'flex', alignItems: 'center', gap: 6,
                    }}>
                      {isActive && (
                        <div style={{
                          width: 6, height: 6, borderRadius: '50%',
                          background: 'rgba(200,40,40,0.85)',
                          boxShadow: '0 0 6px rgba(200,40,40,0.4)',
                        }} />
                      )}
                      {label}
                    </span>
                    <span style={{
                      background: isActive ? 'rgba(200,40,40,0.08)' : 'rgba(110,50,200,0.06)',
                      color: isActive ? 'rgba(190,30,30,0.9)' : (secs > 0 ? 'rgba(55,20,110,0.85)' : 'rgba(110,40,190,0.75)'),
                      border: `1px solid ${isActive ? 'rgba(200,40,40,0.18)' : 'rgba(110,50,200,0.12)'}`,
                      padding: '2px 10px',
                      borderRadius: 100,
                      fontVariantNumeric: 'tabular-nums',
                      fontWeight: 700,
                      fontSize: '0.8rem',
                    }}>
                      {fmtSecs(secs)}
                    </span>
                  </div>
                )
              })}
            </div>
          )}

          {/* Microbreak countdown — only when calibrated */}
          {calibrated && (
            <div style={{
              fontSize: '0.8rem',
              color: 'rgba(0, 140, 185, 0.92)',
              borderTop: '1px solid rgba(130,60,210,0.1)',
              paddingTop: 16,
              fontWeight: 600,
              textAlign: 'center',
              letterSpacing: '0.03em',
            }}>
              {status.in_microbreak
                ? `Break active — ${status.microbreak_countdown}s remaining`
                : `Next break in ${fmtSecs(nextBreakIn)}`}
            </div>
          )}

          {/* Pre-calibration hint */}
          {!calibrated && (
            <div style={{
              fontSize: '0.8rem',
              color: 'rgba(75, 30, 155, 0.72)',
              fontStyle: 'italic',
              fontWeight: 600,
              textAlign: 'center',
            }}>
              Press Calibrate to begin
            </div>
          )}
        </>
      ) : (
        /* Session not running — show ONLY this, nothing else */
        <div style={{
          fontSize: '0.8rem',
          color: 'rgba(75, 30, 155, 0.72)',
          fontStyle: 'italic',
          fontWeight: 600,
          textAlign: 'center',
        }}>
          Press Calibrate to begin
        </div>
      )}
    </div>
  )
}
