import { Link } from 'react-router-dom'
import { Film } from 'lucide-react'
import { ago, api, type Analysis } from '../api'
import { Card, DeleteAnalysis, Empty, PageHead, Status, useFetch } from '../ui'

export default function Analyses() {
  const { data, reload } = useFetch<Analysis[]>(() => api.get('/api/analyses'), [], 5000)
  return <div className="page">
    <PageHead title="Video analyses" sub="Every uploaded video, its processing state and results." actions={<Link className="btn primary" to="/">Upload video</Link>} />
    <Card pad={false}>
      {!data ? <div className="card-b"><div className="skeleton" style={{ height: 160 }} /></div>
        : !data.length ? <Empty title="No analyses yet" icon={<Film size={20} />}>Upload a video from the dashboard.</Empty>
          : <table className="table"><thead><tr><th>Video</th><th>Checks</th><th>Status</th><th>Progress</th><th>Findings</th><th>Review</th><th>Challans</th><th>Uploaded</th><th /></tr></thead>
            <tbody>{data.map(a => <tr key={a.id}>
              <td><Link className="fname" title={a.filename} to={`/analyses/${a.id}`}>{a.filename}</Link></td>
              <td className="small muted">{a.options.checks.join(', ')}</td>
              <td><Status s={a.status} /></td>
              <td style={{ width: 160 }}>{a.status === 'processing' || a.status === 'queued'
                ? <><div className="progress"><i style={{ width: `${a.progress * 100}%` }} /></div><div className="tiny faint">{a.stage}</div></>
                : a.status === 'failed' ? <span className="tiny" style={{ color: 'var(--red)' }}>{a.error}</span> : <span className="faint small">—</span>}</td>
              <td>{a.n_findings}</td><td>{a.n_review || 0}</td><td>{a.n_challans}</td><td className="faint small">{ago(a.created)}</td><td><DeleteAnalysis a={a} onDone={reload} small /></td>
            </tr>)}</tbody></table>}
    </Card>
  </div>
}
