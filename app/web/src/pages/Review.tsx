import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Check, ClipboardCheck, HelpCircle, X } from 'lucide-react'
import { api, fmtTime, offenceText, type Finding } from '../api'
import { Card, Conf, Empty, PageHead, Spinner, useFetch } from '../ui'
import { EvidenceMedia, Explanation, FindingMeta, MoreEvidence, PlateBlock } from './FindingView'

export default function Review() {
  const { id } = useParams(); const nav = useNavigate()
  const { data: q, reload } = useFetch<Finding[]>(() => api.get('/api/findings?status=pending_review'), [])
  const cur = id || q?.[0]?.id
  const { data: f } = useFetch<Finding>(() => cur ? api.get(`/api/findings/${cur}`) : Promise.resolve(null as any), [cur])
  const [note, setNote] = useState(''); const [plate, setPlate] = useState(''); const [busy, setBusy] = useState(false); const [msg, setMsg] = useState('')
  const all = f ? [f.violation, ...(f.offences || []).map(o => o.violation)] : []
  const [charge, setCharge] = useState<string[]>([])
  useEffect(() => { setCharge(all) }, [f?.id])
  useEffect(() => { setNote(''); setPlate(''); setMsg('') }, [cur])
  useEffect(() => { setPlate(f?.evidence?.plate?.candidate || '') }, [f?.id])
  const decide = async (decision: string) => {
    if (!f || busy) return
    setBusy(true)
    try {
      if (decision === 'challan' && !charge.length) { setMsg('Tick at least one offence to charge'); setBusy(false); return }
      const r = await api.post(`/api/findings/${f.id}/review`, { decision, note, plate, offences: charge })
      const next = q?.find(x => x.id !== f.id)
      setMsg(decision === 'challan' ? 'Challan created' : decision === 'reject' ? 'Marked as no violation' : 'Marked as needing more evidence')
      reload()
      if (r.challan_id && !next) nav(`/challans/${r.challan_id}`)
      else nav(next ? `/review/${next.id}` : '/review')
    } catch (e: any) { setMsg(e.message) } finally { setBusy(false) }
  }
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) return
      if (e.key === 'c') decide('challan'); if (e.key === 'r') decide('reject'); if (e.key === 'm') decide('more_evidence')
    }
    addEventListener('keydown', k); return () => removeEventListener('keydown', k)
  })
  return <div className="page">
    <PageHead title="Human review" sub="Cases the AI was not sure about. Your decision is recorded with the evidence." />
    {!q ? <Spinner /> : !q.length && !id ? <Card><Empty title="The queue is empty" icon={<ClipboardCheck size={20} />}>New uncertain cases appear here after each analysis.</Empty></Card>
      : <div className="split" style={{ gridTemplateColumns: '300px minmax(0,1fr)' }}>
        <Card title={`Queue (${q.length})`} pad={false}>
          {q.map(x => <Link key={x.id} to={`/review/${x.id}`} className={'list-item' + (x.id === cur ? ' sel' : '')} style={{ color: 'inherit', textDecoration: 'none' }}>
            <div style={{ minWidth: 0, flex: 1 }}><div className="small" style={{ fontWeight: 500 }}>{offenceText(x)}</div>
              <div className="tiny faint" style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>{x.filename} · {fmtTime(x.t)}</div></div>
            <Conf v={x.confidence} /></Link>)}
        </Card>
        {!f ? <Spinner /> : <div className="grid">
          <div className="grid g2" style={{ alignItems: 'start' }}>
            <Card title={<span>{offenceText(f)} <span className="faint small">· AI prediction: <span className="mono">{f.model_decision}</span></span></span>}><EvidenceMedia f={f} /></Card>
            <div className="grid">
              <Card title="Decision">
                {f.status !== 'pending_review' && f.status !== 'needs_evidence' ? <div className="note">Already decided: {f.status}</div> : <>
                  <PlateBlock f={f} />
                  {!f.evidence?.plate?.plate && <div className="field" style={{ marginTop: 10 }}><label htmlFor="pl">Plate number for the challan (confirm or correct)</label>
                    <input id="pl" className="input mono" placeholder="e.g. AP 03 BK 2645" value={plate} onChange={e => setPlate(e.target.value)} /></div>}
                  {all.length > 1 && <div className="field"><label>Offences to charge (same vehicle, one challan)</label>
                    {all.map(v => <label key={v} className="row small" style={{ gap: 8, fontWeight: 400 }}><input type="checkbox" checked={charge.includes(v)}
                      onChange={e => setCharge(e.target.checked ? all.filter(x => x === v || charge.includes(x)) : charge.filter(x => x !== v))} />{v}</label>)}</div>}
                  <div className="field"><label htmlFor="nt">Note (saved with the decision)</label><textarea id="nt" rows={2} value={note} onChange={e => setNote(e.target.value)} /></div>
                  <div className="grid" style={{ gap: 8 }}>
                    <button className="btn primary block" disabled={busy} onClick={() => decide('challan')}><Check size={15} /> Create challan <kbd style={{ marginLeft: 'auto' }}>C</kbd></button>
                    <button className="btn block" disabled={busy} onClick={() => decide('reject')}><X size={15} /> Reject — no violation <kbd style={{ marginLeft: 'auto' }}>R</kbd></button>
                    <button className="btn block" disabled={busy} onClick={() => decide('more_evidence')}><HelpCircle size={15} /> Needs more evidence <kbd style={{ marginLeft: 'auto' }}>M</kbd></button>
                  </div></>}
                {msg && <div className="alert ok" style={{ marginTop: 12, marginBottom: 0 }}>{msg}</div>}
              </Card>
              <Card><FindingMeta f={f} /></Card>
            </div>
          </div>
          {!!f.merged?.length && <Card><MoreEvidence f={f} /></Card>}
          <Card><Explanation f={f} /></Card>
          <div className="tiny faint">Detected objects and tracks: open the full video in <Link to={`/analyses/${f.analysis_id}`}>the analysis</Link> to step through every frame with the model overlay.</div>
        </div>}
      </div>}
  </div>
}
