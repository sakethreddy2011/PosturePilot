import { BrowserRouter, Routes, Route } from 'react-router-dom'
import CustomCursor from './components/CustomCursor.jsx'
import Landing from './components/Landing.jsx'
import Monitor from './components/Monitor.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <CustomCursor />
      
      {/* Landing-matching animated background (global) */}
      <div className="bg-orbs">
        <div className="orb o1" />
        <div className="orb o2" />
        <div className="orb o3" />
        <div className="orb o4" />
      </div>
      <div className="bg-grid" />

      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Monitor />} />
      </Routes>
    </BrowserRouter>
  )
}
