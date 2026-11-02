import { api } from '../api'
import { Card, PageHead, Spinner, useFetch } from '../ui'

export default function System() {
  const { data } = useFetch<any>(() => api.get('/api/system'), [])
  if (!data) return <div className="page"><Spinner /></div>
  return <div className="page">
    <PageHead title="Models & system" sub="The trained models this installation uses, and what each one needs." />
    <div className="grid g2">
      <Card title="Violation detectors" pad={false}>
        <table className="table"><thead><tr><th>Detector</th><th>Produces</th><th>Needs</th><th>Status</th></tr></thead>
          <tbody>{data.detectors.map((d: any) => <tr key={d.key}><td>{d.title}</td><td className="small">{d.produces.join(', ')}</td>
            <td className="small muted">{d.needs || '—'}</td><td><span className={`badge ${d.available ? 'b-green' : 'b-red'}`}>{d.status}</span></td></tr>)}</tbody></table>
      </Card>
      <div className="grid">
        <Card title="Supporting models"><dl className="kv">
          <dt>Number plates</dt><dd>{data.plate.status}</dd>
          <dt>Emergency vehicles</dt><dd>{data.emergency.model} · exempt: {data.emergency.exempt.join(', ')}</dd>
          <dt>Lanes</dt><dd>{data.lanes.title} — {data.lanes.role}</dd>
        </dl></Card>
        <Card title="How decisions are made"><ul className="reasons small muted">
          <li>Every model tracks each vehicle across many frames and decides once per vehicle — never from a single frame.</li>
          <li>Thresholds were chosen on cameras the models never trained on, favouring no wrong challans.</li>
          <li>Uncertain cases always go to human review. Extra persons on a motorcycle are always reviewed.</li>
          <li>Plates are printed only when read with certainty; otherwise a reviewer may type them from the crop.</li>
        </ul></Card>
      </div>
    </div>
  </div>
}
