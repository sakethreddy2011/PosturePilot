const post = (url) => fetch(url, { method: 'POST' }).then(r => r.json())

export const startCamera   = () => post('/api/start')
export const stopCamera    = () => post('/api/stop')
export const calibrate     = () => post('/api/calibrate')
export const toggleNodes   = () => post('/api/toggle_nodes')
export const endMicrobreak = () => post('/api/end_microbreak')
