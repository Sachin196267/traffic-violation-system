import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ChevronLeft, FlaskConical, Info, Play } from 'lucide-react'
import { api, fmtDate, type Analysis } from '../api'
import { Card, Empty, PageHead, Spinner, useFetch } from '../ui'

type Group = { label: string; vehicles: number; per_hour: number; mix: Record<string, number>; counts: Record<string, number> }
type Row = { controller: string; avg_wait: number; person_wait: number; max_wait: number; amb_wait: number; amb_runs: number
  red_jumps_per_hour: number; box_blocked_pct: number; throughput_per_hour: number; queued_at_end: number }
type SimState = {
  ready: boolean; status: 'none' | 'queued' | 'running' | 'done' | 'failed'; error?: string; started?: number; finished?: number
  measured: { seconds: number; parked_or_single_frame: number; groups: Record<'toward' | 'away' | 'all', Group> } | null
  job?: { rates: number[]; ambPerHour: number; busStops: boolean; cattlePerHour: number; minutes: number; seeds: number[]
    measured: { group: string; vehicles: number; seconds: number; per_hour: number; lanes: number; sim_per_hour: number; capped: boolean } }
  result?: { rows: Row[]; minutes: number; seeds: number[] }
}

const TYPES = ['Bike', 'Auto', 'Car', 'Bus', 'Truck', 'Bicycle']
const ABOUT: Record<string, string> = {
  'Fixed 30 s': 'Fixed rotation, 30 s green each',
  'Webster': 'Fixed-time plan designed from the flows (Webster / IRC:SP:41)',
  'Adaptive v1': 'Queue-pressure adaptive control',
  'PRAVAH': 'Person-weighted, patience-aware adaptive control (this project)',
}

const Sim = () => <span className="badge b-amber" title="Produced by the traffic simulator, not measured at the junction">SIMULATED</span>

function Mix({ g }: { g: Group }) {
  return <div className="row wrap" style={{ gap: 6 }}>{TYPES.filter(t => g.counts[t]).map(t =>
    <span key={t} className="badge b-gray">{t} {Math.round(g.mix[t] * 100)}% ({g.counts[t]})</span>)}</div>
}

export default function SignalSimPage() {
  const { id } = useParams()
  const { data: a } = useFetch<Analysis>(() => api.get(`/api/analyses/${id}`), [id])
  const [poll, setPoll] = useState(0)
  const { data: s, error, reload } = useFetch<SimState>(() => api.get(`/api/analyses/${id}/signal-sim`), [id], poll)
  const [group, setGroup] = useState<'toward' | 'away' | 'all'>('toward')
  const [lanes, setLanes] = useState(3)
  const [other, setOther] = useState([600, 600, 600])
  const [amb, setAmb] = useState(4)
  const [busStops, setBusStops] = useState(true)
  const [cattle, setCattle] = useState(0)
  const [minutes, setMinutes] = useState(20)
  const [msg, setMsg] = useState<string | null>(null)
  const busy = s?.status === 'queued' || s?.status === 'running'
  useEffect(() => { setPoll(busy ? 3000 : 0) }, [busy])
  useEffect(() => {                           // reopen with the inputs of the last run
    const j = s?.job; if (!j) return
    setGroup(j.measured.group as any); setLanes(j.measured.lanes); setOther(j.rates.slice(1)); setAmb(j.ambPerHour)
    setBusStops(j.busStops); setCattle(j.cattlePerHour); setMinutes(j.minutes)
  }, [s?.job?.measured?.vehicles, s?.started])

  if (error) return <div className="page"><Empty title="Could not load the simulation">{error}</Empty></div>
  if (!s || !a) return <div className="page"><Spinner /></div>
  const m = s.measured
  const g = m?.groups[group]
  const simRate = g ? Math.min(2500, Math.round(g.per_hour * 2 / lanes)) : 0
  const run = async () => {
    setMsg(null)
    try { await api.post(`/api/analyses/${id}/signal-sim`, { group, lanes, other, amb_per_hour: amb, bus_stops: busStops, cattle_per_hour: cattle, minutes }); reload(); setPoll(3000) }
    catch (e: any) { setMsg(e.message) }
  }
  const rows = s.result?.rows || []
  const best = (k: keyof Row) => rows.length ? Math.min(...rows.map(r => r[k] as number)) : NaN

  return <div className="page">
    <PageHead title={<span className="row">Signal timing <Sim /></span> as any}
      sub={<>How different signal controllers would handle the traffic measured in <b>{a.filename}</b> · uploaded {fmtDate(a.created)}</>}
      actions={<Link className="btn" to={`/analyses/${id}`}><ChevronLeft size={14} /> Back to the analysis</Link>} />
    <div className="note amber" style={{ marginBottom: 16 }}><FlaskConical size={15} style={{ flex: 'none', marginTop: 2 }} />
      <span>Everything on this page is <b>simulated</b>. The filmed approach's flow and vehicle mix are measured from this video; the other three approaches,
        ambulances and driver behaviour are assumptions of the simulator (Unity, Indian mixed traffic). It estimates how controllers compare — it is not a
        measurement of this junction's signal and never creates findings or challans.</span></div>
    {!s.ready && <div className="alert err">The simulator is not available on this server (Unity or signal_sim/ missing).</div>}
    {!m ? <Empty title="No vehicle tracks in this analysis">Traffic can only be measured from analyses with the shared detection pass.</Empty> :
    <div className="split">
      <div className="grid">
        <Card title="1 · Traffic measured from the video">
          <p className="small muted" style={{ marginTop: 0 }}>{m.seconds.toFixed(0)} s of video. A camera sees both directions — pick the vehicles that drive towards the signal.
            {m.parked_or_single_frame > 0 && ` ${m.parked_or_single_frame} parked or briefly seen vehicles are not counted.`}</p>
          <div className="grid" style={{ gap: 8 }}>{(['toward', 'away', 'all'] as const).map(k => {
            const x = m.groups[k]
            return <label key={k} className={'list-item' + (group === k ? ' sel' : '')} style={{ cursor: 'pointer', alignItems: 'flex-start' }}>
              <input type="radio" name="grp" checked={group === k} onChange={() => setGroup(k)} style={{ marginTop: 3 }} />
              <div style={{ flex: 1 }}><div className="small" style={{ fontWeight: 500 }}>{x.label}: {x.vehicles} vehicles · ~{x.per_hour} / hour</div>
                <div style={{ marginTop: 6 }}><Mix g={x} /></div></div>
            </label>
          })}</div>
          <div className="row wrap" style={{ gap: 16, marginTop: 14 }}>
            <div className="field" style={{ marginBottom: 0 }}><label htmlFor="ln">Lanes on the filmed approach</label>
              <input id="ln" className="input" type="number" min={1} max={8} value={lanes} onChange={e => setLanes(+e.target.value || 1)} style={{ width: 110 }} /></div>
            <div className="small muted" style={{ flex: 1, minWidth: 220 }}>The simulated junction has 2 lanes per approach, so the per-lane flow is kept:
              <b> {g?.per_hour ?? 0} × 2 / {lanes} = {simRate} vehicles/hour</b> enter the simulated North approach{g && g.per_hour * 2 / lanes > 2500 ? ' (capped at 2,500)' : ''}.
              Tracking can split one vehicle into two, so treat the flow as an upper estimate.</div>
          </div>
        </Card>
        <Card title="2 · Assumptions for the rest of the junction">
          <div className="grid g2">{['East', 'South', 'West'].map((n, i) => <div key={n} className="field"><label htmlFor={'o' + i}>{n} approach (vehicles/hour, assumed)</label>
            <input id={'o' + i} className="input" type="number" min={0} max={2500} step={50} value={other[i]} onChange={e => setOther(other.map((v, j) => j === i ? +e.target.value : v))} /></div>)}
            <div className="field"><label htmlFor="amb">Ambulances per hour</label><input id="amb" className="input" type="number" min={0} max={30} value={amb} onChange={e => setAmb(+e.target.value)} /></div>
            <div className="field"><label htmlFor="cat">Stray cattle per hour</label><input id="cat" className="input" type="number" min={0} max={20} value={cattle} onChange={e => setCattle(+e.target.value)} /></div>
            <div className="field"><label htmlFor="min">Simulated minutes per run</label><input id="min" className="input" type="number" min={5} max={30} value={minutes} onChange={e => setMinutes(+e.target.value)} /></div>
          </div>
          <label className="row small"><input type="checkbox" checked={busStops} onChange={e => setBusStops(e.target.checked)} /> Buses and autos stop just after the junction</label>
          {msg && <div className="alert err" style={{ marginTop: 12 }}>{msg}</div>}
          <div className="row" style={{ marginTop: 14, gap: 12 }}>
            <button className="btn primary" disabled={busy || !s.ready || !g || g.vehicles < 5} onClick={run}><Play size={14} /> {busy ? 'Simulating…' : 'Run simulation'}</button>
            <span className="small muted">4 controllers × 3 random seeds on identical traffic · about 1–3 minutes</span>
          </div>
        </Card>
      </div>
      <div className="grid">
        <Card title={<span className="row">3 · Result <Sim /></span>} pad={false}>
          {busy ? <div className="row" style={{ padding: 16 }}><Spinner /><span className="small">Running the simulator… started {s.started ? fmtDate(s.started) : ''}</span></div>
            : s.status === 'failed' ? <div className="alert err" style={{ margin: 16 }}>Simulation failed: {s.error}</div>
            : !rows.length ? <Empty title="Not simulated yet">Set the inputs and run the simulation.</Empty>
            : <>
              <table className="table"><thead><tr><th>Controller</th><th>Avg wait</th><th>Per person</th><th>Worst</th><th>Ambulance</th><th>Red jumps /h</th><th>Vehicles /h</th><th>Left queued</th></tr></thead>
                <tbody>{rows.map(r => <tr key={r.controller}>
                  <td><div style={{ fontWeight: 500 }}>{r.controller}</div><div className="tiny faint">{ABOUT[r.controller]}</div></td>
                  {(['avg_wait', 'person_wait', 'max_wait'] as const).map(k => <td key={k} style={{ fontWeight: r[k] === best(k) ? 700 : 400 }}>{r[k].toFixed(0)} s</td>)}
                  <td>{r.amb_runs ? `${r.amb_wait.toFixed(1)} s` : '–'}</td>
                  <td style={{ fontWeight: r.red_jumps_per_hour === best('red_jumps_per_hour') ? 700 : 400 }}>{r.red_jumps_per_hour.toFixed(0)}</td>
                  <td>{r.throughput_per_hour.toFixed(0)}</td><td>{r.queued_at_end.toFixed(0)}</td>
                </tr>)}</tbody></table>
              {s.job && <div className="small muted" style={{ padding: '12px 16px' }}>
                Inputs: North {s.job.rates[0].toFixed(0)} veh/h from the video ({s.job.measured.vehicles} vehicles in {s.job.measured.seconds.toFixed(0)} s, {s.job.measured.lanes} lanes → 2),
                East {s.job.rates[1]}, South {s.job.rates[2]}, West {s.job.rates[3]} veh/h assumed · {s.job.ambPerHour} ambulances/h · {s.job.minutes} simulated minutes × seeds {s.job.seeds.join(', ')}.
                Bold = best. Waits are time stopped before the stop line; "Ambulance" shows – when no ambulance reached the junction in these runs.</div>}
            </>}
        </Card>
        <Card title="How to read this">
          <div className="note blue"><Info size={15} style={{ flex: 'none', marginTop: 2 }} /><span>
            All four controllers get exactly the same arrivals, so differences come from the signal logic alone. The absolute numbers depend on the
            assumed approaches and on the simulator's driver model (patience before jumping a red, bus-stop dwell, two-wheeler filtering); the
            comparison between controllers is the meaningful part. Open the simulator in Unity to watch the same traffic.</span></div>
        </Card>
      </div>
    </div>}
  </div>
}
