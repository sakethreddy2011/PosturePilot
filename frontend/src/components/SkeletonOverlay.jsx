// SVG skeleton overlay drawn on top of the raw video feed.
// landmark coords are normalised [0,1] from MediaPipe; we scale by videoDims.
const CONNECTIONS = [
  ['left_shoulder', 'right_shoulder'],
  ['left_hip', 'right_hip'],
  // midline
  ['left_eye', 'right_eye'],      // eye pair (we'll use midpoints below)
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
]

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
}

export default function SkeletonOverlay({ landmarks, videoDims, show }) {
  if (!show || !landmarks || !videoDims.width) return null

  const { width: W, height: H } = videoDims

  const px = (lm) => lm ? [lm[0] * W, lm[1] * H] : null

  const nodes = {
    nose:           px(landmarks.nose),
    left_eye:       px(landmarks.left_eye),
    right_eye:      px(landmarks.right_eye),
    left_ear:       px(landmarks.left_ear),
    right_ear:      px(landmarks.right_ear),
    left_shoulder:  px(landmarks.left_shoulder),
    right_shoulder: px(landmarks.right_shoulder),
    left_hip:       px(landmarks.left_hip),
    right_hip:      px(landmarks.right_hip),
  }

  // Computed midpoints for midline
  const eyeMid    = nodes.left_eye && nodes.right_eye    ? midpoint(nodes.left_eye, nodes.right_eye)       : null
  const shMid     = nodes.left_shoulder && nodes.right_shoulder ? midpoint(nodes.left_shoulder, nodes.right_shoulder) : null
  const hipMid    = nodes.left_hip && nodes.right_hip    ? midpoint(nodes.left_hip, nodes.right_hip)       : null

  const lines = [
    // shoulder bar
    [nodes.left_shoulder, nodes.right_shoulder],
    // hip bar
    [nodes.left_hip, nodes.right_hip],
    // midline
    [eyeMid, shMid],
    [shMid, hipMid],
    // sides
    [nodes.left_shoulder, nodes.left_hip],
    [nodes.right_shoulder, nodes.right_hip],
  ]

  const circleNodes = Object.values(nodes).filter(Boolean)

  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      width={W}
      height={H}
      viewBox={`0 0 ${W} ${H}`}
    >
      {lines.map((pair, i) => {
        if (!pair[0] || !pair[1]) return null
        return (
          <line
            key={i}
            x1={pair[0][0]} y1={pair[0][1]}
            x2={pair[1][0]} y2={pair[1][1]}
            stroke="#50C878"
            strokeWidth={2}
            strokeLinecap="round"
          />
        )
      })}
      {circleNodes.map((pt, i) => (
        <circle
          key={i}
          cx={pt[0]} cy={pt[1]}
          r={5}
          fill="#6B8AFF"
          stroke="#3050CC"
          strokeWidth={1.5}
        />
      ))}
    </svg>
  )
}
