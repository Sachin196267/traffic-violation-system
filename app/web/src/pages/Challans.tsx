import { Link, useParams } from 'react-router-dom'
import { Download, FileText } from 'lucide-react'
import { api, fmtDate, fmtTime } from '../api'
import { Card, Conf, Empty, PageHead, Spinner, useFetch } from '../ui'
import { EvidenceMedia, Explanation, MoreEvidence } from './FindingView'

export default function Challans() {
  const { data } = useFetch<any[]>(() => api.get('/api/challans'), [])
  return <div className="page">
    <PageHead title="Challans" sub="Issued automatically for high-confidence findings, or by a reviewer. Each one links to its evidence." />
    <Card pad={false}>
      {!data ? <div className="card-b"><div className="skeleton" style={{ height: 160 }} /></div>
        : !data.length ? <Empty title="No challans yet" icon={<FileText size={20} />}>They appear after an analysis or a review decision.</Empty>
          : <table className="table"><thead><tr><th>Challan</th><th>Violation</th><th>Plate</th><th>Source</th><th>Confidence</th><th>Video · time</th><th>Issued</th><th /></tr></thead>
            <tbody>{data.map(c => <tr key={c.id}>
              <td><Link className="mono" to={`/challans/${c.id}`}>{c.number}</Link></td><td>{(c.offences || [c.violation]).join(' + ')}</td>
              <td>{c.plate ? <span className="plate" style={{ fontSize: 12, padding: '1px 6px' }}>{c.plate}</span> : <span className="faint small">—</span>}</td>
              <td><span className={`badge ${c.source === 'automatic' ? 'b-gray' : 'b-green'}`}>{c.source === 'automatic' ? 'Automatic' : 'Human review'}</span></td>
              <td><Conf v={c.confidence} /></td><td className="small muted">{c.filename} · {fmtTime(c.t)}</td><td className="faint small">{fmtDate(c.created)}</td>
              <td><a className="btn sm" href={`/api/challans/${c.id}/pdf`} aria-label={`Download ${c.number} as PDF`}><Download size={13} /> PDF</a></td>
            </tr>)}</tbody></table>}
    </Card>
  </div>
}

export function ChallanDetail() {
  const { id } = useParams()
  const { data } = useFetch<any>(() => api.get(`/api/challans/${id}`), [id])
  if (!data) return <div className="page"><Spinner /></div>
  const { challan: c, finding: f, analysis: a, review: r, reviewer } = data
  return <div className="page">
    <PageHead title={`Challan ${c.number}`} sub={`${(c.offences || [f.violation]).join(' + ')} · issued ${fmtDate(c.created)} · ${c.source === 'automatic' ? 'automatic (high confidence)' : 'after human review'}`}
      actions={<a className="btn primary" href={`/api/challans/${c.id}/pdf`}><Download size={14} /> Download PDF</a>} />
    <div className="split">
      <div className="grid"><Card title="Evidence"><EvidenceMedia f={f} /></Card>{!!f.merged?.length && <Card><MoreEvidence f={f} /></Card>}<Card><Explanation f={{ ...f, challan: c }} /></Card></div>
      <div className="grid">
        <Card title="Challan details"><dl className="kv">
          <dt>Challan ID</dt><dd className="mono">{c.number}</dd><dt>Offence{(c.offences || []).length > 1 ? 's' : ''}</dt><dd>{(c.offences || [f.violation]).join(' + ')}</dd>
          <dt>Video</dt><dd><Link to={`/analyses/${a.id}`}>{a.filename}</Link> at {fmtTime(f.t)}</dd>
          <dt>Vehicle / track</dt><dd>{f.vehicle || '—'} · <span className="mono">{f.track}</span></dd>
          <dt>AI confidence</dt><dd><Conf v={f.confidence} /></dd>
          <dt>Plate on challan</dt><dd>{c.plate ? <span className="plate">{c.plate}</span> : <span className="faint">Not confirmed</span>}</dd>
          <dt>Evidence SHA-256</dt><dd className="mono tiny faint">{c.evidence_hash}</dd>
        </dl></Card>
        {r && <Card title="Human review"><dl className="kv"><dt>Reviewer</dt><dd>{reviewer?.name}</dd><dt>Decided</dt><dd>{fmtDate(r.created)}</dd><dt>Note</dt><dd>{r.note || '—'}</dd></dl></Card>}
      </div>
    </div>
  </div>
}
