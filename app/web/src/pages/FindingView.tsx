import { useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, Info, Layers, ShieldCheck } from 'lucide-react'
import { api, fmtTime, offenceText, type Finding } from '../api'
import { Conf, PlateView, Status } from '../ui'

const DET: Record<string, string> = {
  helmet: 'Helmet model — YOLO11 head classifier, per-rider vote across frames',
  triple: 'Triple-riding model — rider-count classifier + heads on the seat, frame agreement',
  redlight: 'Signal-state model + stop-line crossing (hidden-Markov smoothing of the signal colour)',
  speed: 'Speed estimation — calibrated camera geometry with a 90 % range',
}

export function EvidenceMedia({ f }: { f: Finding }) {
  const ev = f.evidence || {}
  const [clip, setClip] = useState<'annotated' | 'original'>(ev.clip_annotated ? 'annotated' : 'original')
  const [zoom, setZoom] = useState<string | null>(null)
  const src = (n: string) => api.file(f.analysis_id, `evidence/${f.id}/${n}`)
  const clipName = clip === 'annotated' ? ev.clip_annotated : ev.clip
  return <div className="grid" style={{ gap: 12 }}>
    {clipName && <div>
      <div className="player"><video key={clipName} src={src(clipName)} controls muted playsInline loop autoPlay /></div>
      <div className="ctrl">
        <span className="small muted">Segment {fmtTime(f.t_start)} – {fmtTime(f.t_end)}</span>
        <span style={{ flex: 1 }} />
        {ev.clip_annotated && <div className="seg">
          <button className={clip === 'annotated' ? 'on' : ''} onClick={() => setClip('annotated')}>Model overlay</button>
          <button className={clip === 'original' ? 'on' : ''} onClick={() => setClip('original')}>Original</button>
        </div>}
      </div>
    </div>}
    <div className="thumbs">
      {(ev.frames || []).map((fr: any) => <figure key={fr.raw}>
        <img src={src(fr.boxed || fr.raw)} alt={`Evidence frame at ${fr.t} s`} onClick={() => setZoom(src(fr.boxed || fr.raw))} />
        <figcaption>{fmtTime(fr.t)}{fr.boxed ? ' · detection box' : ''}</figcaption>
      </figure>)}
    </div>
    {zoom && <div className="lightbox" onClick={() => setZoom(null)}><img src={zoom} alt="Evidence frame" /></div>}
  </div>
}

export function Explanation({ f }: { f: Finding }) {
  const ev = f.evidence || {}
  const why = f.status === 'exempt' ? 'The vehicle was recognised as an emergency vehicle, so no challan is issued.'
    : f.route === 'review' ? 'The model found a possible violation but its own decision rule marked it uncertain, so a person must decide.'
      : 'The model’s multi-frame decision passed its high-confidence threshold (set on held-out cameras), so a challan was issued automatically.'
  return <div className="grid" style={{ gap: 14 }}>
    <div className="row"><ShieldCheck size={16} className="muted" /><h3>Explainable AI</h3></div>
    <dl className="kv">
      <dt>What was detected</dt><dd>{offenceText(f)} · {f.vehicle || 'vehicle'} track <span className="mono">{f.track}</span></dd>
      <dt>Model</dt><dd>{DET[f.detector] || f.detector}</dd>
      <dt>Model decision</dt><dd><span className="mono">{f.model_decision}</span></dd>
      <dt>Confidence</dt><dd><Conf v={f.confidence} /></dd>
      <dt>Supporting frames</dt><dd>{(ev.frames || []).map((x: any) => fmtTime(x.t)).join(', ') || '—'}</dd>
    </dl>
    <div><div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>Why</div><p className="small muted" style={{ margin: '0 0 6px' }}>{why}</p>
      <ul className="reasons small">{f.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul></div>
    {f.uncertain_reasons?.length > 0 && <div className="note amber"><AlertTriangle size={15} style={{ flex: 'none', marginTop: 2 }} />
      <div><b>Why it needs review</b><ul className="reasons">{f.uncertain_reasons.map((r, i) => <li key={i}>{r}</li>)}</ul></div></div>}
    {f.emergency && <div className="note blue"><Info size={15} /> Emergency vehicle: {f.emergency}. Emergency vehicles are never challaned.</div>}
    <VehicleCase f={f} />
  </div>
}

const at = (x: { analysis_id: string; id: string }) => `/analyses/${x.analysis_id}?f=${x.id}`

/** one challan or review per vehicle: the offences in this case and the detections merged into it */
export function VehicleCase({ f }: { f: Finding }) {
  const charged: string[] | undefined = f.challan?.offences
  if (f.merged_into) return <div className="note blue"><Layers size={15} style={{ flex: 'none', marginTop: 2 }} />
    <div>This detection is part of another case for the same vehicle: <Link to={at(f.merged_into)}>{f.merged_into.violation}</Link> at {fmtTime(f.merged_into.t)}
      {f.merged_into.analysis_id !== f.analysis_id && <> in <b>{f.merged_into.filename}</b></>} ({f.merged_into.status.replace('_', ' ')}). It is never challaned separately.</div></div>
  if (!f.offences?.length && !f.merged?.length) return null
  return <div className="grid" style={{ gap: 8 }}>
    {!!f.offences?.length && <div><div className="small" style={{ fontWeight: 600, marginBottom: 4 }}>Offences of this vehicle (one case)</div>
      <ul className="reasons small">
        <li>{f.violation} — <Conf v={f.confidence} />{charged && (charged.includes(f.violation) ? ' · charged' : ' · not charged')}</li>
        {f.offences.map(o => <li key={o.id}><Link to={at(o)}>{o.violation}</Link> at {fmtTime(o.t)} — <Conf v={o.confidence} />
          {charged ? (charged.includes(o.violation) ? ' · charged' : ' · not certain, not charged') : o.certain ? ' · certain' : ' · not certain'}</li>)}
      </ul></div>}
  </div>
}

/** one case per vehicle, all of its evidence: every other detection merged into this case */
export function MoreEvidence({ f }: { f: Finding }) {
  const m = f.merged || []
  const [play, setPlay] = useState<string | null>(null)
  const [zoom, setZoom] = useState<string | null>(null)
  if (!m.length) return null
  return <div className="grid" style={{ gap: 14 }}>
    <div className="row"><Layers size={16} className="muted" /><h3>More evidence of this vehicle</h3>
      <span className="faint small">{m.length} other detection{m.length > 1 ? 's' : ''} · same case, never a separate challan</span></div>
    {m.map(x => {
      const ev = x.evidence || {}; const src = (n: string) => api.file(x.analysis_id, `evidence/${x.id}/${n}`)
      const why = x.why
      const crop = (ev.frames || []).find((fr: any) => fr.crop)?.crop
      return <div key={x.id} className="more-ev">
        <div className="row small" style={{ gap: 8, flexWrap: 'wrap' }}>
          <b>{x.violation}</b><span className="muted">at {fmtTime(x.t)}</span><Conf v={x.confidence} />
          <span style={{ flex: 1 }} />
          {(ev.clip_annotated || ev.clip) && <button className="btn ghost sm" onClick={() => setPlay(play === x.id ? null : x.id)}>{play === x.id ? 'Show frames' : 'Play clip'}</button>}
        </div>
        {why && <div className="tiny faint" style={{ margin: '2px 0 8px' }}>{why}</div>}
        {play === x.id ? <EvidenceMedia f={x} /> : <div className="thumbs" style={{ gridTemplateColumns: `repeat(${(ev.frames || []).length + (crop ? 1 : 0)}, 1fr)` }}>
          {(ev.frames || []).map((fr: any) => <figure key={fr.raw}><img src={src(fr.boxed || fr.raw)} alt={`${x.violation} at ${fr.t} s`} loading="lazy"
            onClick={() => setZoom(src(fr.boxed || fr.raw))} /><figcaption>{fmtTime(fr.t)}</figcaption></figure>)}
          {crop && <figure><img src={src(crop)} alt="Close-up" loading="lazy" onClick={() => setZoom(src(crop))} /><figcaption>Close-up</figcaption></figure>}
        </div>}
      </div>
    })}
    {zoom && <div className="lightbox" onClick={() => setZoom(null)}><img src={zoom} alt="Evidence" /></div>}
  </div>
}

export function PlateBlock({ f }: { f: Finding }) {
  const p = f.evidence?.plate
  return <div>
    <div className="row" style={{ marginBottom: 8 }}><h3>Number plate</h3><span className="faint tiny">plate detector + Indian plate reader</span></div>
    <div className="row gap16">
      {p?.crop && <img src={api.file(f.analysis_id, `evidence/${f.id}/${p.crop}`)} alt="Plate crop" style={{ height: 44, borderRadius: 4, border: '1px solid var(--border)' }} />}
      <div>{p?.plate ? <PlateView p={p} /> : p?.candidate ? <span className="row"><span className="plate" style={{ borderStyle: 'dashed' }}>{p.candidate}</span><span className="badge b-amber">unconfirmed reading</span></span> : <PlateView p={p} />}
        {p && !p.plate && <div className="tiny faint" style={{ marginTop: 4, maxWidth: 360 }}>
          {p.candidate ? 'The reader was not certain. Compare with the crop and confirm or correct it below.' : (p.reasons || [])[0]}</div>}</div>
    </div>
  </div>
}

export function FindingMeta({ f }: { f: Finding }) {
  return <dl className="kv">
    <dt>Status</dt><dd><Status s={f.status} /></dd>
    <dt>Timestamp</dt><dd>{fmtTime(f.t)} in <span className="muted">{f.filename}</span></dd>
    <dt>Track ID</dt><dd className="mono">{f.track}</dd>
    <dt>Evidence hash</dt><dd className="mono tiny faint">{f.evidence?.sha256?.slice(0, 32)}…</dd>
  </dl>
}
