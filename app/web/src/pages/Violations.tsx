import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ShieldAlert } from 'lucide-react'
import { ago, api, fmtTime, offenceText, type Finding } from '../api'
import { Card, Conf, Empty, PageHead, Status, useFetch } from '../ui'

const FILTERS = [['', 'All'], ['challan', 'Challan issued'], ['pending_review', 'Needs review'], ['needs_evidence', 'More evidence'], ['rejected', 'Rejected'], ['exempt', 'Exempt'], ['duplicate', 'Duplicates']]

export default function Violations() {
  const [st, setSt] = useState('')
  const { data } = useFetch<Finding[]>(() => api.get('/api/findings' + (st ? `?status=${st}` : '')), [st])
  return <div className="page">
    <PageHead title="Violations" sub="Every finding from every analysis, with its decision path." />
    <div className="seg" style={{ marginBottom: 12 }}>{FILTERS.map(([k, l]) => <button key={k} className={st === k ? 'on' : ''} onClick={() => setSt(k)}>{l}</button>)}</div>
    <Card pad={false}>
      {!data ? <div className="card-b"><div className="skeleton" style={{ height: 160 }} /></div>
        : !data.length ? <Empty title="No violations" icon={<ShieldAlert size={20} />}>Nothing matches this filter.</Empty>
          : <table className="table"><thead><tr><th>Evidence</th><th>Violation</th><th>Video · time</th><th>Track</th><th>Confidence</th><th>Status</th><th>Found</th></tr></thead>
            <tbody>{data.map(f => { const fr = f.evidence?.frames?.[1] || f.evidence?.frames?.[0]; return <tr key={f.id}>
              <td>{fr && <img src={api.file(f.analysis_id, `evidence/${f.id}/${fr.boxed || fr.raw}`)} alt="" style={{ width: 72, height: 44, objectFit: 'cover', borderRadius: 4 }} />}</td>
              <td><Link to={f.status === 'pending_review' ? `/review/${f.id}` : `/analyses/${f.analysis_id}?f=${f.id}`}>{offenceText(f)}</Link></td>
              <td className="small muted">{f.filename} · {fmtTime(f.t)}</td><td className="mono">{f.track}</td><td><Conf v={f.confidence} /></td><td><Status s={f.status} /></td><td className="faint small">{ago(f.created)}</td>
            </tr> })}</tbody></table>}
    </Card>
  </div>
}
