import { useState, useEffect } from 'react'

export function useStatus(intervalMs = 500) {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    let active = true

    const poll = async () => {
      try {
        const res = await fetch('/api/status')
        if (res.ok && active) {
          setStatus(await res.json())
        }
      } catch {
        // backend not yet up — silently ignore
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [intervalMs])

  return status
}
