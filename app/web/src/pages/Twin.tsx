import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Activity, Info, Pause, Play } from 'lucide-react'
import { api, fmtTime, type Analysis } from '../api'
import { Card, Empty, PageHead, Spinner, useFetch } from '../ui'
import Twin3D from './Twin3D'

const TYPE_COL: Record<string, string> = { car: '#2f6fe4', motorcycle: '#0ea5a4', autorickshaw: '#d99a06', bus: '#7c3aed', truck: '#9a5b2b', bicycle: '#16a34a' }
const ROUTE_COL: Record<string, string> = { confirmed: '#e5484d', review: '#f59e0b', exempt: '#3b82f6' }

type Twin = { meta: any; filename: string; scene: string | null; tracks: { id: number; type: string; path: number[][]; plate: string | null }[]
  lanes: any; lane_files: any; findings: any[]; source: string | null }

function at(path: number[][], t: number) {
  // box at time t (linear between samples); null outside the track
  if (!path.length || t < path[0][0] - 0.1 || t > path[path.length - 1][0] + 0.1) return null
  let i = path.findIndex(p => p[0] >= t)
  if (i <= 0) return path[Math.max(0, i)].slice(1)
  const a = path[i - 1], b = path[i], k = (t - a[0]) / Math.max(1e-3, b[0] - a[0])
  return a.slice(1).map((v, j) => v + (b[j + 1] - v) * k)
}

export function TwinView({ id }: { id: string }) {
  const { data } = useFetch<Twin>(() => api.get(`/api/analyses/${id}/twin`), [id])
  const cv = useRef<HTMLCanvasElement>(null)
  const img = useRef<HTMLImageElement | null>(null)
  const [ready, setReady] = useState(0)
  const [t, setT] = useState(0); const [play, setPlay] = useState(false); const [speed, setSpeed] = useState(1)
  const [opt, setOpt] = useState({ trails: true, labels: true, scene: true, onlyViol: false })
  const [mode, setMode] = useState<'3d' | '2d'>('3d')
  const [view, setView] = useState('orbit')
  const started = useRef(false)
  const { data: lanes } = useFetch<any>(() => data?.lane_files?.lanes ? fetch(api.file(id, data.lane_files.lanes)).then(r => r.json()) : Promise.resolve(null), [data?.lane_files?.lanes])
  const dur = data?.meta?.duration || Math.max(1, ...(data?.tracks || []).map(x => x.path.at(-1)?.[0] || 0))
  const W = data?.meta?.width || 1280, H = data?.meta?.height || 720
  const hot = useMemo(() => (data?.findings || []).filter(f => f.box), [data])
  useEffect(() => {
    if (!data?.tracks?.length || started.current) return
    started.current = true
    let best = 0, bt = 0
    for (let x = 0; x < dur; x += 0.5) { const n = data.tracks.filter(tr => at(tr.path, x)).length; if (n > best) { best = n; bt = x } }
    setT(bt)
  }, [data])
  useEffect(() => {
    if (!data?.scene) return
    const im = new Image(); im.src = api.file(id, data.scene); im.onload = () => { img.current = im; setReady(r => r + 1) }
  }, [data?.scene])
  useEffect(() => {
    if (!play) return
    let last = performance.now(), raf = 0
    const tick = (now: number) => { setT(x => { const n = x + (now - last) / 1000 * speed; return n > dur ? 0 : n }); last = now; raf = requestAnimationFrame(tick) }
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf)
  }, [play, speed, dur])
  useEffect(() => {
    const c = cv.current; if (!c || !data) return
    const g = c.getContext('2d')!; const s = c.width / W
    g.clearRect(0, 0, c.width, c.height)
    if (opt.scene && img.current) { g.globalAlpha = 0.55; g.drawImage(img.current, 0, 0, c.width, c.height); g.globalAlpha = 1 }
    else { g.fillStyle = '#eef4fc'; g.fillRect(0, 0, c.width, c.height) }
    const active = hot.filter(f => Math.abs(f.t - t) < 1.2)
    for (const tr of data.tracks) {
      const b = at(tr.path, t); if (!b) continue
      const viol = active.find(f => { const fb = f.box; return fb && !(fb[2] < b[0] || fb[0] > b[2] || fb[3] < b[1] || fb[1] > b[3]) })
      if (opt.onlyViol && !viol) continue
      const col = viol ? ROUTE_COL[viol.route] : TYPE_COL[tr.type] || '#64748b'
      if (opt.trails) {
        g.strokeStyle = col + '88'; g.lineWidth = 2; g.beginPath()
        tr.path.filter(p => p[0] <= t && p[0] > t - 3).forEach((p, i) => { const x = (p[1] + p[3]) / 2 * s, y = p[4] * s; i ? g.lineTo(x, y) : g.moveTo(x, y) })
        g.stroke()
      }
      g.fillStyle = col + (viol ? '55' : '2a'); g.strokeStyle = col; g.lineWidth = viol ? 3 : 1.5
      g.beginPath(); g.roundRect(b[0] * s, b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s, 4); g.fill(); g.stroke()
      if (opt.labels) {
        const txt = viol ? viol.violation : `${tr.type} #${tr.id}${tr.plate ? ' · ' + tr.plate : ''}`
        g.font = '600 11px Inter, sans-serif'; const w = g.measureText(txt).width + 8
        g.fillStyle = col; g.fillRect(b[0] * s, b[1] * s - 16, w, 15); g.fillStyle = '#fff'; g.fillText(txt, b[0] * s + 4, b[1] * s - 5)
      }
    }
  }, [t, data, opt, hot, ready])
  if (!data) return <Spinner />
  if (!data.tracks.length) return <Empty title="No vehicle tracks for this analysis" icon={<Activity size={20} />}>No vehicle tracks were stored for this analysis. Run a new analysis to see the reconstruction.</Empty>
  const inView = data.tracks.filter(tr => at(tr.path, t)).length
  const tog = (k: keyof typeof opt) => setOpt(o => ({ ...o, [k]: !o[k] }))
  return <div className="split">
    <Card title="Scene reconstruction" right={<div className="seg"><button className={mode === '3d' ? 'on' : ''} onClick={() => setMode('3d')}>3D model</button><button className={mode === '2d' ? 'on' : ''} onClick={() => setMode('2d')}>Camera view</button></div>}>
      {mode === '3d' && <Twin3D data={data} lanes={lanes} t={t} opt={opt} view={view} />}
      {mode === '3d' && <div className="row" style={{ marginTop: 8 }}><span className="tiny faint">View</span><div className="seg">{[['orbit', '3D orbit'], ['camera', 'From the camera'], ['bird', "Bird's-eye"]].map(([k, l]) => <button key={k} className={view === k ? 'on' : ''} onClick={() => setView(k)}>{l}</button>)}</div><span className="tiny faint">drag to rotate · scroll to zoom · right-drag to pan</span></div>}
      <canvas ref={cv} width={960} height={Math.round(960 * H / W)} style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)', background: '#eef4fc', display: mode === '2d' ? 'block' : 'none' }} />
      <div className="tl" onClick={e => { const r = e.currentTarget.getBoundingClientRect(); setT((e.clientX - r.left) / r.width * dur) }}>
        {data.findings.map(f => <span key={f.id} className="mk" style={{ left: `${f.t / dur * 100}%`, background: ROUTE_COL[f.route] }} title={`${f.violation} · ${fmtTime(f.t)}`} />)}
        <span className="ph" style={{ left: `${t / dur * 100}%` }} />
      </div>
      <div className="ctrl">
        <button className="btn sm" onClick={() => setPlay(!play)} aria-label={play ? 'Pause' : 'Play'}>{play ? <Pause size={14} /> : <Play size={14} />}</button>
        <div className="seg">{[0.5, 1, 2, 4].map(x => <button key={x} className={speed === x ? 'on' : ''} onClick={() => setSpeed(x)}>{x}×</button>)}</div>
        <span className="mono muted">{fmtTime(t)} / {fmtTime(dur)}</span><span style={{ flex: 1 }} />
        {(mode === '3d' ? (['labels', 'onlyViol'] as const) : (['scene', 'trails', 'labels', 'onlyViol'] as const)).map(k => <label key={k} className="small row" style={{ gap: 4 }}><input type="checkbox" checked={opt[k]} onChange={() => tog(k)} />
          {{ scene: 'Scene', trails: 'Trails', labels: 'Labels', onlyViol: 'Violations only' }[k]}</label>)}
      </div>
    </Card>
    <div className="grid">
      <Card title="At this moment"><dl className="kv"><dt>Vehicles in view</dt><dd>{inView}</dd><dt>Tracked in total</dt><dd>{data.tracks.length}</dd>
        <dt>Lanes estimated</dt><dd>{data.lanes?.ok ? 'yes — drawn on the scene' : 'no (too little moving traffic)'}</dd><dt>Findings</dt><dd>{data.findings.length}</dd></dl></Card>
      <Card title="Legend">
        <div className="grid" style={{ gap: 6 }}>{Object.entries(TYPE_COL).map(([k, c]) => <span key={k} className="row small"><i style={{ width: 12, height: 12, borderRadius: 3, background: c }} />{k}</span>)}</div>
        <div className="divider" />
        <div className="grid" style={{ gap: 6 }}>{Object.entries(ROUTE_COL).map(([k, c]) => <span key={k} className="row small"><i style={{ width: 12, height: 12, borderRadius: 3, background: c }} />{{ confirmed: 'Violation · challan', review: 'Violation · review', exempt: 'Emergency · exempt' }[k]}</span>)}</div>
      </Card>
      <Link className="btn" to={`/lanes/${id}`}>Lane detection for this road</Link>
      <div className="note blue"><Info size={15} style={{ flex: 'none', marginTop: 2 }} /><span>Reconstructed from real tracks: {data.source}. The 3D view places vehicles on the road using the camera's horizon from lane detection (flat road, approximate scale). Drag to orbit, scroll to zoom.</span></div>
      <Link className="btn" to={`/analyses/${id}`}>Open the full analysis</Link>
    </div>
  </div>
}

export default function TwinPage() {
  const { id } = useParams(); const nav = useNavigate()
  const { data: an } = useFetch<Analysis[]>(() => api.get('/api/analyses'), [])
  const done = (an || []).filter(a => a.status === 'done')
  useEffect(() => { if (!id && done.length) nav(`/twin/${done[0].id}`, { replace: true }) }, [id, done.length])
  return <div className="page">
    <PageHead title="Digital twin" sub="A live reconstruction of the scene: every tracked vehicle, the estimated lanes and each violation in time."
      actions={done.length > 0 && <select value={id} onChange={e => nav(`/twin/${e.target.value}`)} style={{ width: 260 }}>{done.map(a => <option key={a.id} value={a.id}>{a.filename}</option>)}</select>} />
    {!an ? <Spinner /> : !done.length ? <Card><Empty title="No finished analyses yet">Upload a video on the dashboard.</Empty></Card> : id && <TwinView id={id} />}
  </div>
}
