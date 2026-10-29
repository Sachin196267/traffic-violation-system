import { useState } from 'react'
import { FolderOpen, X } from 'lucide-react'
import { api, fmtTime, offenceText, type Finding } from '../api'
import { Card, Empty, PageHead, Status, useFetch } from '../ui'
import { EvidenceMedia, Explanation, FindingMeta } from './FindingView'

export default function Evidence() {
  const { data } = useFetch<Finding[]>(() => api.get('/api/findings'), [])
  const [type, setType] = useState(''); const [sel, setSel] = useState<Finding | null>(null)
  const types = [...new Set((data || []).map(f => f.violation))]
  const rows = (data || []).filter(f => !type || f.violation === type)
  return <div className="page">
    <PageHead title="Evidence" sub="Every finding keeps its frames, clips, boxes, track ID, confidence and a SHA-256 fingerprint." />
    <div className="seg" style={{ marginBottom: 12 }}>{['', ...types].map(k => <button key={k} className={type === k ? 'on' : ''} onClick={() => setType(k)}>{k || 'All'}</button>)}</div>
    {!data ? <div className="skeleton" style={{ height: 200 }} /> : !rows.length ? <Card><Empty title="No evidence yet" icon={<FolderOpen size={20} />}>Evidence appears after an analysis.</Empty></Card>
      : <div className="gallery">{rows.map(f => { const fr = f.evidence?.frames?.[1] || f.evidence?.frames?.[0]; return <button key={f.id} className="card gcell" onClick={() => setSel(f)}>
        {fr && <img src={api.file(f.analysis_id, `evidence/${f.id}/${fr.boxed || fr.raw}`)} alt={f.violation} loading="lazy" />}
        <div className="gmeta"><div className="small" style={{ fontWeight: 600 }}>{offenceText(f)}</div><div className="tiny faint">{f.filename} · {fmtTime(f.t)} · <span className="mono">{f.track}</span></div>
          <div style={{ marginTop: 6 }}><Status s={f.status} /></div></div></button> })}</div>}
    {sel && <div className="drawer-bg" onClick={() => setSel(null)}><aside className="drawer" onClick={e => e.stopPropagation()}>
      <div className="row" style={{ marginBottom: 12 }}><h2>{offenceText(sel)}</h2><Status s={sel.status} /><button className="btn ghost sm" style={{ marginLeft: 'auto' }} onClick={() => setSel(null)} aria-label="Close"><X size={15} /></button></div>
      <div className="grid" style={{ gap: 16 }}><EvidenceMedia f={sel} /><FindingMeta f={sel} /><Explanation f={sel} /></div>
    </aside></div>}
  </div>
}
