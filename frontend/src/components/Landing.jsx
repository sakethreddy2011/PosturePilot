import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import './Landing.css'

export default function Landing() {
  useEffect(() => {
    const navbar = document.getElementById('navbar')
    const handleScroll = () => {
      if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 10)
    }
    window.addEventListener('scroll', handleScroll)

    const sections = document.querySelectorAll('section[id]')
    const links    = document.querySelectorAll('.nav-link')

    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(l => l.classList.remove('active'))
          const active = document.querySelector(`.nav-link[href="#${e.target.id}"]`)
          if (active) active.classList.add('active')
        }
      })
    }, { threshold: 0.5 })

    sections.forEach(s => obs.observe(s))

    return () => {
      window.removeEventListener('scroll', handleScroll)
      sections.forEach(s => obs.unobserve(s))
      obs.disconnect()
    }
  }, [])

  return (
    <div className="landing-page">
      <nav id="navbar">
        <span className="nav-logo">PosturePilot</span>
        <ul className="nav-links">
          <li><a href="#detect" className="nav-link">What it detects</a></li>
          <li className="nav-cta"><Link to="/app" className="nav-link"><span>Open App ↗</span></Link></li>
        </ul>
      </nav>

      {/* Hero */}
      <section id="hero">
        <div className="section-tag">Real-time ergonomics</div>
        <h2>Your posture, <span>perfected</span><br/>in real time.</h2>
        <p className="section-body">
          AI-powered webcam monitoring that catches bad posture the moment it happens — no wearables, no sensors, just your camera and computer vision.
        </p>
        <div className="hero-btns">
          <Link to="/app" className="btn-primary"><span>Launch App</span></Link>
        </div>
      </section>

      <div className="divider"></div>

      {/* What it detects */}
      <section id="detect">
        <div className="section-tag">What we detect</div>
        <h2>Three defects.<br/><span>One camera.</span></h2>
        <p className="section-body">
          The system identifies the most common biomechanical deviations from a single webcam feed, calibrated to your unique anatomy.
        </p>
        <div className="cards-grid three">
          <div className="card-dark">
            <div className="card-dark-img-wrap">
              <img src="/hunching.jpg" alt="Hunching posture" className="card-dark-img" />
            </div>
            <div className="card-dark-title">Hunching (Kyphosis)</div>
          </div>
          <div className="card-dark">
            <div className="card-dark-img-wrap">
              <img src="/lateral-lean.jpg" alt="Lateral lean posture" className="card-dark-img" />
            </div>
            <div className="card-dark-title">Lateral Lean</div>
          </div>
          <div className="card-dark">
            <div className="card-dark-img-wrap">
              <img src="/hip-sliding.jpg" alt="Hip sliding posture" className="card-dark-img" />
            </div>
            <div className="card-dark-title">Hip Sliding</div>
          </div>
        </div>
      </section>

      <div className="divider"></div>

      {/* CTA */}
      <section id="start">
        <div className="section-tag">Get started</div>
        <h2>Start monitoring.<br/><span>Start healing.</span></h2>
        <p className="section-body">
          Your next work session is a calibration click away. No installation beyond what you already have.
        </p>
        <Link to="/app" className="btn-primary" style={{ marginTop: 36 }}><span>Launch App →</span></Link>
      </section>
    </div>
  )
}
