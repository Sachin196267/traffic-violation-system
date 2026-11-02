import { useState } from 'react'
import { Camera, ChevronDown, Clock, Moon, MonitorPlay, Ruler, Scissors, TrafficCone, X } from 'lucide-react'

// Guidance comes from what the models were tested on (held-out cameras) and where they failed.
const TIPS = [
  { i: <Camera size={16} />, t: 'Fixed camera', d: 'CCTV, pole or tripod. Hand-held or moving footage breaks vehicle tracking.' },
  { i: <Scissors size={16} />, t: 'One continuous shot', d: 'No cuts or compilations — every cut restarts tracking and short shots cannot be judged.' },
  { i: <MonitorPlay size={16} />, t: '720p or better, sharp', d: '1080p is ideal. Riders’ heads should be at least ~40 px tall; plates ~20 px to be read.' },
  { i: <Ruler size={16} />, t: 'Angle: 4–8 m high, looking along the road', d: 'Tilted down 20–45°. Avoid straight top-down (drone) views and very far vehicles.' },
  { i: <Moon size={16} />, t: 'Daylight works best', d: 'At night lamps glare and fewer vehicles are detected; more cases go to review.' },
  { i: <Clock size={16} />, t: '30 seconds or longer', d: 'Each vehicle is judged over many frames, so it needs to stay in view for a few seconds.' },
  { i: <TrafficCone size={16} />, t: 'Red light is automatic; speed needs a setup', d: 'Red light: checked on every video, no setup. Best is a fixed camera with the signal in view through a red → green change (1–3 min): clear jumps then become challans. If the light stays red for the whole clip (e.g. filmed from a car waiting at the red), vehicles that drive past it go to human review. No signal in view: nothing is reported. Speed: one known road length (calibration file).' },
]

export default function VideoGuide() {
  const key = 'tvp-guide-collapsed'
  const [closed, setClosed] = useState(() => { try { return localStorage.getItem(key) === '1' } catch { return false } })
  const set = (v: boolean) => { setClosed(v); try { localStorage.setItem(key, v ? '1' : '0') } catch { /* ignore */ } }
  if (closed) return <button className="btn sm" style={{ marginBottom: 16 }} onClick={() => set(false)}><ChevronDown size={14} /> What makes a good video?</button>
  return <section className="card fade-in" style={{ marginBottom: 16, background: 'linear-gradient(180deg,#ffffff,#f6faff)' }}>
    <div className="card-h"><h2>Before you upload — what makes a good video</h2>
      <span className="faint small">Better footage means more confident decisions and fewer cases for review</span>
      <button className="btn ghost sm right" onClick={() => set(true)} aria-label="Hide guide"><X size={14} /></button></div>
    <div className="card-b guide">
      {TIPS.map(x => <div key={x.t} className="tip"><div className="tip-i">{x.i}</div><div><div className="tip-t">{x.t}</div><div className="tip-d">{x.d}</div></div></div>)}
    </div>
  </section>
}
