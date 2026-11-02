import { useRef, useState, type DragEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowRight, Film, UploadCloud } from 'lucide-react'
import { ago, api, offenceText, type Analysis, type Finding } from '../api'
import { Card, Conf, DeleteAnalysis, Empty, PageHead, Stat, Status, useFetch } from '../ui'
import VideoGuide from './VideoGuide'
import { getDefaults } from './Settings'

// every video gets every check; speed also needs a calibration file and a limit (km/h can't be measured without them)
const ALL_CHECKS = ['helmet', 'triple', 'redlight']

export function Upload() {
  const nav = useNavigate()
  const inp = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [over, setOver] = useState(false)
  const [maxSec, setMaxSec] = useState(getDefaults().max_sec || '120')
  const [calib, setCalib] = useState<File | null>(null)
  const [limit, setLimit] = useState('')
  const [busy, setBusy] = useState(false); const [pct, setPct] = useState(0); const [err, setErr] = useState('')
  const drop = (e: DragEvent) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) setFile(f) }
  const start = () => {
    if (!file) return
    if (!!calib !== !!limit) return setErr('For the speed check, give both a calibration file and a speed limit (or neither).')
    setErr(''); setBusy(true)
    const fd = new FormData()
    fd.append('file', file)
    const checks = calib && limit ? [...ALL_CHECKS, 'speed'] : ALL_CHECKS
    fd.append('options', JSON.stringify({ checks, max_sec: Number(maxSec) || 0, lanes: true, plates: true, speed_limit: Number(limit) || undefined }))
    if (calib) fd.append('speed_calib', calib)
    const x = new XMLHttpRequest()      // XHR for upload progress
    x.open('POST', '/api/analyses'); x.setRequestHeader('X-TVP', '1')
    x.upload.onprogress = e => e.lengthComputable && setPct(e.loaded / e.total)
    x.onload = () => {
      setBusy(false)
      if (x.status < 300) nav(`/analyses/${JSON.parse(x.responseText).id}`)
      else setErr((() => { try { return JSON.parse(x.responseText).detail } catch { return 'Upload failed' } })())
    }
    x.onerror = () => { setBusy(false); setErr('Upload failed — check the connection') }
    x.send(fd)
  }
  return <Card title="Analyse a video">
    <div className={'drop' + (over ? ' over' : '')} onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={drop}
      onClick={() => inp.current?.click()} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && inp.current?.click()}>
      <div className="ico"><UploadCloud size={20} /></div>
      {file ? <><b>{file.name}</b><div className="small muted">{(file.size / 1e6).toFixed(1)} MB · click to choose another</div></>
        : <><b>Drop a video here</b><div className="small muted">or click to browse · MP4, MOV, AVI, MKV, WEBM</div></>}
      <input ref={inp} type="file" accept="video/*" hidden onChange={e => e.target.files?.[0] && setFile(e.target.files[0])} />
    </div>
    <div className="small muted" style={{ marginTop: 14 }}>Every video is checked for <b>no helmet</b>, <b>triple riding</b> and <b>red-light jumping</b>, with number-plate reading,
      emergency-vehicle exemption and lane estimation. Nothing to choose.</div>
    <details style={{ marginTop: 10 }}>
      <summary className="small">Optional: also check over-speeding (needs a camera calibration file and a speed limit)</summary>
      <div className="grid g2" style={{ marginTop: 10 }}>
        <div className="field"><label>Speed calibration (JSON)</label><input type="file" accept=".json" onChange={e => setCalib(e.target.files?.[0] || null)} /></div>
        <div className="field"><label>Speed limit (km/h)</label><input className="input" value={limit} onChange={e => setLimit(e.target.value)} inputMode="numeric" /></div>
      </div>
    </details>
    <div className="row" style={{ marginTop: 14 }}>
      <label className="small muted" htmlFor="ms">Analyse the first</label>
      <select id="ms" value={maxSec} onChange={e => setMaxSec(e.target.value)} style={{ width: 130 }}>
        <option value="30">30 seconds</option><option value="60">1 minute</option><option value="120">2 minutes</option><option value="300">5 minutes</option><option value="0">whole video</option>
      </select>
      <span style={{ flex: 1 }} />
      <button className="btn primary" disabled={!file || busy} onClick={start}>
        {busy ? `Uploading ${Math.round(pct * 100)}%` : <>Start analysis <ArrowRight size={14} /></>}</button>
    </div>
    <div className="tiny faint" style={{ marginTop: 8 }}>Lanes are context only and never create a challan.</div>
    {err && <div className="alert err" style={{ marginTop: 12, marginBottom: 0 }}>{err}</div>}
  </Card>
}

export default function Dashboard() {
  const { data: st } = useFetch<any>(() => api.get('/api/stats'), [], 10000)
  const { data: an, reload } = useFetch<Analysis[]>(() => api.get('/api/analyses'), [], 5000)
  const { data: rq } = useFetch<Finding[]>(() => api.get('/api/findings?status=pending_review'), [], 10000)
  const n = (k: string) => st ? st[k] : '—'
  return <><div className="page">
    <PageHead title="Dashboard" sub="Upload recorded traffic video; confirmed violations become challans, uncertain ones wait for review." />
    <VideoGuide />
    <div className="grid g4" style={{ marginBottom: 16 }}>
      <Stat label="Videos analysed" value={n('analyses_done')} detail={st?.processing ? `${st.processing} processing` : 'none processing'} />
      <Stat label="Violations found" value={n('findings')} detail={st ? `${st.confirmed} high-confidence` : ''} />
      <Stat label="Challans issued" value={n('challans')} detail={st ? `${st.challans_reviewed} after human review` : ''} />
      <Stat label="Waiting for review" value={n('pending_review')} detail={st ? `${st.exempt} emergency vehicles exempted` : ''} />
    </div>
    <div className="split">
      <div className="grid">
        <Upload />
        <Card title="Recent analyses" right={<Link to="/analyses" className="small">All analyses</Link>} pad={false}>
          {!an ? <div className="card-b"><div className="skeleton" style={{ height: 80 }} /></div>
            : !an.length ? <Empty title="No videos yet" icon={<Film size={20} />}>Upload a video to see results here.</Empty>
              : <table className="table"><thead><tr><th>Video</th><th>Status</th><th>Findings</th><th>Challans</th><th>Uploaded</th><th /></tr></thead>
                <tbody>{an.slice(0, 6).map(a => <tr key={a.id} className="click" onClick={() => location.assign(`/analyses/${a.id}`)}>
                  <td><Link className="fname" title={a.filename} to={`/analyses/${a.id}`}>{a.filename}</Link></td>
                  <td><Status s={a.status} /> {a.status === 'processing' && <span className="tiny faint">{a.stage}</span>}</td>
                  <td>{a.n_findings}{a.n_review ? <span className="tiny faint"> · {a.n_review} to review</span> : ''}</td><td>{a.n_challans}</td>
                  <td className="faint small">{ago(a.created)}</td><td onClick={e => e.stopPropagation()}><DeleteAnalysis a={a} onDone={reload} small /></td></tr>)}</tbody></table>}
        </Card>
      </div>
      <Card title="Human review queue" right={<Link to="/review" className="small">Open queue</Link>} pad={false}>
        {!rq ? <div className="card-b"><div className="skeleton" style={{ height: 120 }} /></div>
          : !rq.length ? <Empty title="Nothing to review">Uncertain cases appear here.</Empty>
            : rq.slice(0, 8).map(f => <Link key={f.id} to={`/review/${f.id}`} className="list-item" style={{ color: 'inherit', textDecoration: 'none' }}>
              {f.evidence?.frames?.[1] && <img src={api.file(f.analysis_id, `evidence/${f.id}/${f.evidence.frames[1].boxed || f.evidence.frames[1].raw}`)} alt="" style={{ width: 64, height: 40, objectFit: 'cover', borderRadius: 4 }} />}
              <div style={{ minWidth: 0, flex: 1 }}><div className="small" style={{ fontWeight: 500 }}>{offenceText(f)}</div>
                <div className="tiny faint" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{f.uncertain_reasons?.[0]}</div></div>
              <Conf v={f.confidence} /></Link>)}
      </Card>
    </div>
  </div></>
}
