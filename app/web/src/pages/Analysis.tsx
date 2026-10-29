import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Download, FileText, FlaskConical, Info, Pause, Play, SkipBack, SkipForward, X } from 'lucide-react'
import { api, fmtDate, fmtTime, offenceText, type Analysis, type Finding } from '../api'
import { Card, Conf, DeleteAnalysis, Empty, PageHead, Spinner, Status, useFetch } from '../ui'
import { EvidenceMedia, Explanation, FindingMeta, MoreEvidence } from './FindingView'

const SRC_LABEL: Record<string, string> = { playable: 'Original', helmet: 'Helmet overlay', triple: 'Triple-riding overlay', redlight: 'Signal & stop line', speed: 'Speed overlay', lanes: 'Lane structure' }
const MARK: Record<string, string> = { confirmed: 'var(--red)', review: 'var(--amber)', exempt: 'var(--blue)' }

function Viewer({ a, sel, onPick }: { a: Analysis; sel: Finding | null; onPick: (f: Finding) => void }) {
  const art = a.meta?.artefacts || {}
  const sources = [['playable', 'playable.mp4'], ...Object.entries(art).filter(([, v]) => typeof v === "string").map(([k, v]) => [k, v as string])]
  if (art.lanes && a.meta?.artefacts?.lanes_summary?.ok) sources.push(['lanes', 'lanes_playable.mp4'])
  const [src, setSrc] = useState('playable')
  const v = useRef<HTMLVideoElement>(null)
  const [t, setT] = useState(0); const [dur, setDur] = useState(a.meta?.duration || 1); const [playing, setPlaying] = useState(false)
  const fps = a.meta?.fps || 25
  useEffect(() => { if (sel && v.current) { v.current.currentTime = sel.t; v.current.pause() } }, [sel?.id])
  const step = (n: number) => { const el = v.current; if (!el) return; el.pause(); el.currentTime = Math.max(0, el.currentTime + n / fps) }
  const file = sources.find(s => s[0] === src)?.[1] || 'playable.mp4'
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return
      if (e.key === ',') step(-1); if (e.key === '.') step(1)
      if (e.key === ' ') { e.preventDefault(); v.current?.paused ? v.current?.play() : v.current?.pause() }
    }
    addEventListener('keydown', k); return () => removeEventListener('keydown', k)
  })
  return <Card title="Video & digital twin" right={<div className="seg">{sources.map(([k]) => <button key={k} className={src === k ? 'on' : ''} onClick={() => setSrc(k)}>{SRC_LABEL[k] || k}</button>)}</div>}>
    <div className="player"><video ref={v} key={file} src={api.file(a.id, file)} playsInline muted preload="metadata"
      onTimeUpdate={e => setT(e.currentTarget.currentTime)} onLoadedMetadata={e => { setDur(e.currentTarget.duration); if (t) e.currentTarget.currentTime = t }}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} /></div>
    <div className="tl" onClick={e => { const r = e.currentTarget.getBoundingClientRect(); if (v.current) v.current.currentTime = (e.clientX - r.left) / r.width * dur }}
      role="slider" aria-label="Timeline" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={t}>
      {(a.findings || []).map(f => <span key={f.id} className="mk" title={`${f.violation} · ${fmtTime(f.t)}`} style={{ left: `${f.t / dur * 100}%`, background: MARK[f.route] }}
        onClick={e => { e.stopPropagation(); onPick(f) }} />)}
      <span className="ph" style={{ left: `${t / dur * 100}%` }} />
    </div>
    <div className="ctrl">
      <button className="btn sm" onClick={() => step(-Math.round(fps))} aria-label="Back 1 second"><SkipBack size={14} /></button>
      <button className="btn sm" onClick={() => step(-1)} aria-label="Previous frame"><ChevronLeft size={14} /></button>
      <button className="btn sm" onClick={() => playing ? v.current?.pause() : v.current?.play()} aria-label={playing ? 'Pause' : 'Play'}>{playing ? <Pause size={14} /> : <Play size={14} />}</button>
      <button className="btn sm" onClick={() => step(1)} aria-label="Next frame"><ChevronRight size={14} /></button>
      <button className="btn sm" onClick={() => step(Math.round(fps))} aria-label="Forward 1 second"><SkipForward size={14} /></button>
      <span className="mono muted">{fmtTime(t)} / {fmtTime(dur)} · frame {Math.round(t * fps)}</span>
      <span style={{ flex: 1 }} />
      <span className="tiny faint"><kbd>,</kbd> <kbd>.</kbd> frame · <kbd>space</kbd> play</span>
    </div>
    {src === 'lanes' || src === 'playable' ? null : <div className="tiny faint" style={{ marginTop: 6 }}>Overlay videos are rendered by the model at its sampling rate; frame numbers refer to the original video.</div>}
  </Card>
}

function Detail({ f: row, ch, onClose }: { f: Finding; ch?: any; onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const { data: full } = useFetch<Finding>(() => api.get(`/api/findings/${row.id}`), [row.id])
  const f = full?.id === row.id ? full : row             // with the other detections of this vehicle
  useEffect(() => { ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, [f.id])
  return <div ref={ref}><Card title={<span className="row">{offenceText(f)} <Status s={f.status} /></span>} right={<button className="btn ghost sm" onClick={onClose} aria-label="Close"><X size={14} /></button>}>
    <div className="grid" style={{ gap: 16 }}>
      {ch && <div className="challan-bar">
        <div><div className="tiny faint">Challan</div><div className="mono" style={{ fontWeight: 600 }}>{ch.number}</div></div>
        <div><div className="tiny faint">Plate</div>{ch.plate ? <span className="plate" style={{ fontSize: 12, padding: '1px 6px' }}>{ch.plate}</span> : <span className="small faint">Not confirmed</span>}</div>
        <div><div className="tiny faint">Issued</div><div className="small">{ch.source === 'automatic' ? 'Automatically' : 'After review'}</div></div>
        <span style={{ flex: 1 }} />
        <Link className="btn sm" to={`/challans/${ch.id}`}>Open challan</Link>
        <a className="btn primary sm" href={`/api/challans/${ch.id}/pdf`}><Download size={13} /> PDF</a>
      </div>}
      <EvidenceMedia f={f} />
      <MoreEvidence f={f} />
      <FindingMeta f={{ ...f }} />
      <div className="divider" style={{ margin: 0 }} />
      <Explanation f={f} />
      {f.status === 'pending_review' && <Link className="btn primary" to={`/review/${f.id}`}>Review this case</Link>}
    </div>
  </Card></div>
}

export default function AnalysisPage() {
  const { id } = useParams(); const nav = useNavigate()
  const [poll, setPoll] = useState(3000)
  const { data: a, error } = useFetch<Analysis>(() => api.get(`/api/analyses/${id}`), [id], poll)
  const [sel, setSel] = useState<Finding | null>(null)
  const [tab, setTab] = useState<'challan' | 'review' | 'all'>('challan')
  const [params] = useSearchParams()
  useEffect(() => {                  // ?f=<finding> opens that detection (links from merged cases)
    const x = a?.findings?.find(y => y.id === params.get('f'))
    if (x && sel?.id !== x.id) { setSel(x); setTab('all') }
  }, [a?.id, params.get('f')])
  useEffect(() => { if (a && (a.status === 'failed' || (a.status === 'done' && !a.meta?.lanes_pending))) setPoll(0) }, [a?.status, a?.meta?.lanes_pending])
  if (error) return <div className="page"><Empty title="Could not load this analysis">{error}</Empty></div>
  if (!a) return <div className="page"><Spinner /></div>
  const busy = a.status === 'queued' || a.status === 'processing'
  const F = a.findings || []
  const lanes = a.meta?.artefacts?.lanes_summary
  const chBy: Record<string, any> = Object.fromEntries((a.challans || []).map((c: any) => [c.finding_id, c]))
  const groups = { challan: F.filter(f => chBy[f.id]), review: F.filter(f => f.status === 'pending_review' || f.status === 'needs_evidence'), all: F }
  const shown = groups[tab]
  return <div className="page">
    <PageHead title={a.filename} sub={<>Uploaded {fmtDate(a.created)} · checks: {a.options.checks.join(', ')}{a.meta?.duration ? ` · ${a.meta.duration.toFixed(0)} s, ${a.meta.width}×${a.meta.height}` : ''}</>}
      actions={<><Status s={a.status} /><DeleteAnalysis a={{ ...a, n_findings: a.findings?.length, n_challans: a.challans?.length }} onDone={() => nav('/analyses')} /></>} />
    {busy && <Card><div className="row" style={{ marginBottom: 10 }}><Spinner /><b>{a.stage}</b><span className="faint small" style={{ marginLeft: 'auto' }}>{Math.round(a.progress * 100)}%</span></div>
      <div className="progress"><i style={{ width: `${Math.max(3, a.progress * 100)}%` }} /></div>
      <div className="row small muted" style={{ marginTop: 10, gap: 16 }}>
        <span>Elapsed <b>{Math.max(0, Math.round((Date.now() / 1000 - a.created) / 60))} min</b></span>
        {a.meta?.duration && (() => { const v = Math.min(a.meta.duration, a.options.max_sec || 1e9); const lo = Math.max(1, Math.round(v * 0.4 / 60)), hi = Math.max(1, Math.round(v * 1.5 / 60))
          return <span>Estimated total <b>{lo === hi ? `~${lo}` : `${lo}–${hi}`} min</b> (quiet roads are faster; very busy junctions take longer)</span> })()}
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>Every vehicle is tracked across frames and each model decides once per vehicle. You can leave this page — processing continues and the result appears in Video analyses.</p></Card>}
    {a.status === 'failed' && <div className="alert err">Processing failed: {a.error}</div>}
    {a.status === 'done' && <div className="split">
      <div className="grid">
        <Viewer a={a} sel={sel} onPick={setSel} />
        {sel && <Detail f={sel} ch={chBy[sel.id]} onClose={() => setSel(null)} />}
      </div>
      <div className="grid">
        <Card title="Results" right={<span className="faint small">click a result to see its evidence</span>} pad={false}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)' }}><div className="seg">
            <button className={tab === 'challan' ? 'on' : ''} onClick={() => setTab('challan')}>Challans ({groups.challan.length})</button>
            <button className={tab === 'review' ? 'on' : ''} onClick={() => setTab('review')}>Needs review ({groups.review.length})</button>
            <button className={tab === 'all' ? 'on' : ''} onClick={() => setTab('all')}>All ({F.length})</button></div></div>
          {!shown.length ? <Empty title={tab === 'challan' ? 'No challans from this video' : tab === 'review' ? 'Nothing waiting for review' : 'No violations found'} icon={<FileText size={20} />}>
              {tab === 'challan' && F.length ? 'Uncertain cases are under “Needs review”.' : 'The selected checks found nothing to report here.'}</Empty>
            : shown.map(f => <div key={f.id} className={'list-item' + (sel?.id === f.id ? ' sel' : '')} onClick={() => setSel(f)} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && setSel(f)}>
              {f.evidence?.frames?.[1] && <img src={api.file(a.id, `evidence/${f.id}/${f.evidence.frames[1].boxed || f.evidence.frames[1].raw}`)} alt="" style={{ width: 56, height: 36, objectFit: 'cover', borderRadius: 4, flex: 'none' }} />}
              <div style={{ minWidth: 0, flex: 1 }}><div className="small" style={{ fontWeight: 500 }}>{offenceText(f)}</div>
                <div className="tiny faint">{chBy[f.id] ? <span className="mono">{chBy[f.id].number}</span> : f.track} · {fmtTime(f.t)}</div></div>
              <div style={{ textAlign: 'right' }}><Status s={f.status} /><div style={{ marginTop: 4 }}><Conf v={f.confidence} /></div></div>
            </div>)}
        </Card>
        <Card title="Lane structure">
          <div className="note blue" style={{ marginBottom: 10 }}><Info size={15} style={{ flex: 'none', marginTop: 2 }} />
            <span>Lanes are estimated from how traffic moves. They are shown for context and the digital twin only — lane position never creates a challan.</span></div>
          {a.meta?.lanes_pending ? <div className="row small muted"><Spinner /> Estimating the lanes of this road — the results above are already final.</div> : lanes?.ok && a.meta.artefacts.lanes?.overlay ? <img src={api.file(a.id, a.meta.artefacts.lanes.overlay)} alt="Estimated lanes" style={{ width: '100%', borderRadius: 6 }} />
            : <p className="small muted" style={{ margin: 0 }}>No lane structure could be estimated for this video (for example, too few moving vehicles or a moving camera).</p>}
        </Card>
        <Card title={<span className="row">Signal timing <span className="badge b-amber">SIMULATED</span></span>}>
          <p className="small muted" style={{ marginTop: 0 }}>Use the traffic measured in this video to compare signal controllers (fixed-time, Webster, adaptive, PRAVAH) in the junction simulator.</p>
          <Link className="btn" to={`/signal/${a.id}`}><FlaskConical size={14} /> Simulate signal timing</Link>
        </Card>
        {a.meta?.notes?.length > 0 && <Card title="Not checked"><ul className="reasons small muted">{a.meta.notes.map((n: string, i: number) => <li key={i}>{n}</li>)}</ul></Card>}
      </div>
    </div>}
  </div>
}
