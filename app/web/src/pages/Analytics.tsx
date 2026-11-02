import { BarChart3 } from 'lucide-react'
import { api } from '../api'
import { Card, Empty, PageHead, Spinner, Stat, useFetch } from '../ui'

function Bars({ items, color = 'var(--accent-2)' }: { items: [string, number][]; color?: string }) {
  const max = Math.max(1, ...items.map(x => x[1]))
  return <div className="bars">{items.map(([k, v]) => <div key={k} className="bar-row"><span className="small">{k}</span>
    <div className="bar-track"><i style={{ width: `${v / max * 100}%`, background: color }} /></div><span className="mono small">{v}</span></div>)}</div>
}

const ST: Record<string, string> = { challan: 'Challan issued', pending_review: 'Waiting for review', rejected: 'Rejected by reviewer', needs_evidence: 'Needs more evidence', exempt: 'Emergency · exempt', duplicate: 'Duplicate · merged', confirmed: 'Confirmed' }

export default function Analytics() {
  const { data: d } = useFetch<any>(() => api.get('/api/analytics'), [])
  if (!d) return <div className="page"><Spinner /></div>
  const empty = !d.findings
  const hist: number[] = d.confidence_hist
  const hmax = Math.max(1, ...hist)
  return <div className="page">
    <PageHead title="Analytics" sub="Computed from this installation’s own analyses and reviews — no sample data." />
    <div className="grid g4" style={{ marginBottom: 16 }}>
      <Stat label="Videos processed" value={d.analyses_done} detail={`${d.analyses} uploaded`} />
      <Stat label="Violations found" value={d.findings} detail={`${d.confirmed} high-confidence`} />
      <Stat label="Human-reviewed" value={Object.values(d.reviews as Record<string, number>).reduce((a, b) => a + b, 0)} detail={`${d.reviews.reject || 0} rejected · ${d.reviews.challan || 0} challaned`} />
      <Stat label="Challans" value={d.challans} detail={`${d.challans_reviewed} after review`} />
    </div>
    {empty ? <Card><Empty title="No data yet" icon={<BarChart3 size={20} />}>Charts fill in as videos are analysed.</Empty></Card> : <div className="grid g2">
      <Card title="Violation types"><Bars items={Object.entries(d.by_type)} /></Card>
      <Card title="Outcome of every finding"><Bars items={Object.entries(d.by_status).map(([k, v]) => [ST[k] || k, v as number])} color="#7aa7ee" /></Card>
      <Card title="Model confidence distribution">
        <div className="hist">{hist.map((v, i) => <div key={i} className="hcol" title={`${i * 10}–${i * 10 + 10} %: ${v}`}><i style={{ height: `${v / hmax * 100}%` }} /><span className="tiny faint">{i * 10}</span></div>)}</div>
        <div className="tiny faint" style={{ marginTop: 6 }}>Confidence (%) reported by each model’s multi-frame decision.</div>
      </Card>
      <Card title="Processing" pad={false}>
        {!d.processing.length ? <Empty title="No finished analyses" /> : <table className="table"><thead><tr><th>Video</th><th>Video length</th><th>Processing time</th></tr></thead>
          <tbody>{d.processing.map((p: any, i: number) => <tr key={i}><td className="small">{p.filename}</td><td className="small">{p.video_s ? `${p.video_s.toFixed(0)} s` : '—'}</td><td className="small">{Math.round(p.processing_s / 60)} min</td></tr>)}</tbody></table>}
      </Card>
      <Card title="Findings by model"><Bars items={Object.entries(d.by_detector)} color="#5bb5b0" /></Card>
    </div>}
  </div>
}
