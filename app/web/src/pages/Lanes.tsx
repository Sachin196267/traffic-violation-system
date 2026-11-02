import { useEffect } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Info, Route } from 'lucide-react'
import { api, type Analysis } from '../api'
import { Card, Empty, PageHead, Spinner, useFetch } from '../ui'

export default function Lanes() {
  const { id } = useParams(); const nav = useNavigate()
  const { data: an } = useFetch<Analysis[]>(() => api.get('/api/analyses'), [])
  const done = (an || []).filter(a => a.status === 'done')
  useEffect(() => { if (!id && done.length) nav(`/lanes/${done[0].id}`, { replace: true }) }, [id, done.length])
  const { data: a } = useFetch<Analysis>(() => id ? api.get(`/api/analyses/${id}`) : Promise.resolve(null as any), [id], 5000)
  const art = a?.meta?.artefacts || {}; const L = art.lanes || {}; const ok = art.lanes_summary?.ok
  return <div className="page">
    <PageHead title="Lane detection" sub="Lanes of each road, learned from how vehicles actually moved — paint markings are ignored."
      actions={done.length > 0 && <select value={id} onChange={e => nav(`/lanes/${e.target.value}`)} style={{ width: 260 }}>{done.map(x => <option key={x.id} value={x.id}>{x.filename}</option>)}</select>} />
    <div className="note amber" style={{ marginBottom: 16 }}><Info size={15} style={{ flex: 'none', marginTop: 2 }} />
      <span>Indian roads often have faded or ignored lane markings, so lanes here are an estimate for positioning and context. <b>No challan is ever issued because of lane position.</b></span></div>
    {!an ? <Spinner /> : !done.length ? <Card><Empty title="No finished analyses yet" icon={<Route size={20} />}>Upload a video on the dashboard.</Empty></Card>
      : !a ? <Spinner /> : a.meta?.lanes_pending ? <Card><Empty title="Estimating lanes…" icon={<Route size={20} />}>This runs right after the violation results and takes about a minute.</Empty></Card> : !ok ? <Card><Empty title="No lanes could be estimated for this video" icon={<Route size={20} />}>Too little moving traffic in the first 30 seconds, or the camera moves.</Empty></Card>
        : <div className="split">
          <Card title={`Estimated lanes · ${a.filename}`}>
            <img src={api.file(a.id, L.background_lanes || L.overlay)} alt="Lanes estimated for this road" style={{ width: '100%', borderRadius: 8 }} />
            {L.video && <><div className="small muted" style={{ margin: '14px 0 6px' }}>Lanes with live traffic</div>
              <div className="player"><video src={api.file(a.id, 'lanes_playable.mp4')} controls muted playsInline loop /></div></>}
          </Card>
          <div className="grid">
            <Card title="How lanes are found"><ul className="reasons small muted" style={{ margin: 0 }}>
              <li>Every vehicle is tracked for the first 30 seconds of the video.</li>
              <li>Their paths are grouped into traffic streams; each stream becomes a lane, using perspective so far and near parts match.</li>
              <li>Camera movement and scene cuts are detected; only a steady segment is used.</li>
            </ul></Card>
            <Card title="Used for"><ul className="reasons small muted" style={{ margin: 0 }}>
              <li>Positioning vehicles in the digital twin</li><li>Context in evidence review</li><li><b>Never</b> as the reason for a challan</li></ul></Card>
            <Link className="btn" to={`/twin/${a.id}`}>Open digital twin</Link>
          </div>
        </div>}
  </div>
}
