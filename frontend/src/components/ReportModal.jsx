export default function ReportModal({ report, onClose }) {
  if (!report) return null

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 400,
      background: 'rgba(0,0,0,0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}
      onClick={onClose}
    >
      <div
        style={{
          background: '#12141a',
          border: '1px solid #2a2d38',
          borderRadius: 12,
          padding: '28px 32px',
          maxWidth: 720,
          width: '90%',
          maxHeight: '80vh',
          overflow: 'auto',
        }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ fontSize: '1rem', fontWeight: 700, color: '#c0c8e0' }}>Session Report</div>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
            {report.pdf_report && (
              <a
                href={report.pdf_report.startsWith('data:') ? report.pdf_report : `/api/reports/${report.pdf_report}`}
                download={`posture_report_${new Date().getFullYear()}${String(new Date().getMonth()+1).padStart(2,'0')}${String(new Date().getDate()).padStart(2,'0')}_${String(new Date().getHours()).padStart(2,'0')}${String(new Date().getMinutes()).padStart(2,'0')}${String(new Date().getSeconds()).padStart(2,'0')}.pdf`}
                style={{
                  background: 'rgba(120,55,200,0.15)',
                  color: '#c0c8e0',
                  border: '1px solid rgba(120,55,200,0.3)',
                  padding: '6px 16px',
                  borderRadius: 100,
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  textDecoration: 'none',
                  transition: 'all 0.2s',
                  cursor: 'pointer'
                }}
                onMouseOver={e => { e.currentTarget.style.background = 'rgba(120,55,200,0.25)' }}
                onMouseOut={e => { e.currentTarget.style.background = 'rgba(120,55,200,0.15)' }}
              >
                ↓ Export PDF
              </a>
            )}
            <button
              onClick={onClose}
              style={{
                background: 'none', border: 'none', color: '#666',
                cursor: 'pointer', fontSize: '1.2rem',
              }}
            >✕</button>
          </div>
        </div>
        <pre style={{
          fontFamily: 'monospace',
          fontSize: '0.75rem',
          color: '#a0a8c0',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          lineHeight: 1.6,
          marginBottom: report.screenshots && report.screenshots.length > 0 ? 24 : 0
        }}>
          {report.text || report}
        </pre>

        {report.screenshots && report.screenshots.length > 0 && (
          <div>
            <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#c0c8e0', marginBottom: 12 }}>
              Postural Deviation Snapshots
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
              {report.screenshots.map((item, i) => {
                const isObject = typeof item === 'object' && item !== null;
                const src = isObject ? item.data : `/api/reports/${item}`;
                const label = isObject ? item.label : item.split('_').slice(1, -2).join(' ').toUpperCase();
                
                return (
                  <div key={i} style={{ borderRadius: 8, overflow: 'hidden', border: '1px solid #2a2d38' }}>
                    <img
                      src={src}
                      alt="Posture snapshot"
                      style={{ display: 'block', width: '100%', height: 'auto', objectFit: 'cover' }}
                    />
                    <div style={{ padding: '8px', background: '#1a1d26', fontSize: '0.7rem', color: '#8a8d98', textAlign: 'center' }}>
                      {label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
