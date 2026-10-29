import { useEffect, useState, type ReactNode } from 'react'
import { Inbox, Trash2 } from 'lucide-react'
import { api } from './api'
import type { Finding } from './api'

export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return <div className="page-head"><div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>{actions && <div className="actions">{actions}</div>}</div>
}

export function Card({ title, right, children, pad = true }: { title?: ReactNode; right?: ReactNode; children: ReactNode; pad?: boolean }) {
  return <section className="card">{title && <div className="card-h"><h2>{title}</h2>{right && <div className="right">{right}</div>}</div>}
    {pad ? <div className="card-b">{children}</div> : children}</section>
}

export function Stat({ label, value, detail }: { label: string; value: ReactNode; detail?: ReactNode }) {
  return <div className="card stat"><div className="l">{label}</div><div className="v">{value}</div>{detail && <div className="d">{detail}</div>}</div>
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return <div className="empty"><div className="ico">{icon || <Inbox size={20} />}</div><h3>{title}</h3>{children && <div className="small">{children}</div>}</div>
}

export const Spinner = () => <div className="spin" role="status" aria-label="Loading" />

export function Conf({ v }: { v: number }) {
  return <div className="conf" title="Model confidence"><div className="bar"><i style={{ width: `${Math.round(v * 100)}%` }} /></div><span>{Math.round(v * 100)}%</span></div>
}

const STATUS: Record<string, [string, string]> = {
  challan: ['Challan issued', 'b-red'], confirmed: ['Confirmed', 'b-red'], pending_review: ['Needs review', 'b-amber'],
  needs_evidence: ['More evidence', 'b-amber'], rejected: ['Rejected', 'b-gray'], exempt: ['Emergency · exempt', 'b-blue'],
  done: ['Done', 'b-green'], processing: ['Processing', 'b-amber'], queued: ['Queued', 'b-gray'], failed: ['Failed', 'b-red'],
  issued: ['Issued', 'b-red'], cancelled: ['Cancelled', 'b-gray'], duplicate: ['Duplicate · merged', 'b-gray'],
}
export function Status({ s }: { s: string }) {
  const [t, c] = STATUS[s] || [s, 'b-gray']
  return <span className={`badge ${c}`}>{t}</span>
}

export function PlateView({ p }: { p: any }) {
  if (!p) return <span className="faint">—</span>
  if (p.plate) return <span className="row"><span className="plate">{p.plate}</span><span className="faint small">{Math.round((p.confidence || 0) * 100)}%</span></span>
  const t: Record<string, string> = { 'NOT SURE': 'Not confirmed', UNREADABLE: 'Unreadable', 'NOT LINKED': 'Not linked', 'NOT AVAILABLE': 'Unavailable', 'NOT CHECKED': 'Not checked' }
  return <span className="faint small">{t[p.decision] || p.decision}</span>
}

export function useFetch<T>(fn: () => Promise<T>, deps: any[] = [], pollMs?: number) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    let live = true
    fn().then(d => live && (setData(d), setError(null))).catch(e => live && setError(e.message))
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])
  useEffect(() => {
    if (!pollMs) return
    const t = setInterval(() => setTick(x => x + 1), pollMs)
    return () => clearInterval(t)
  }, [pollMs])
  return { data, error, reload: () => setTick(x => x + 1) }
}

export const violationTone = (f: Finding) => f.status === 'exempt' ? 'b-blue' : f.route === 'review' ? 'b-amber' : 'b-red'

export function DeleteAnalysis({ a, onDone, small }: { a: { id: string; filename: string; status: string; n_challans?: number; n_findings?: number }; onDone: () => void; small?: boolean }) {
  const [open, setOpen] = useState(false); const [busy, setBusy] = useState(false); const [err, setErr] = useState('')
  const run = async () => { setBusy(true); try { await api.del(`/api/analyses/${a.id}`); setOpen(false); onDone() } catch (e: any) { setErr(e.message) } finally { setBusy(false) } }
  const live = a.status === 'processing' || a.status === 'queued'
  return <>
    <button className={`btn danger ${small ? 'sm' : ''}`} onClick={e => { e.stopPropagation(); setOpen(true) }} aria-label={`Delete ${a.filename}`}><Trash2 size={small ? 13 : 14} />{!small && ' Delete'}</button>
    {open && <div className="drawer-bg" style={{ alignItems: 'center', justifyContent: 'center' }} onClick={() => !busy && setOpen(false)}>
      <div className="card fade-in" style={{ width: 420, padding: 20 }} onClick={e => e.stopPropagation()} role="dialog" aria-modal>
        <h2 style={{ marginBottom: 8 }}>Delete this analysis?</h2>
        <p className="small muted" style={{ marginTop: 0 }}>{live ? 'Processing will be stopped. ' : ''}This permanently deletes <b>{a.filename}</b>, its evidence,
          {a.n_findings != null ? ` ${a.n_findings} findings` : ' all findings'}{a.n_challans != null ? ` and ${a.n_challans} challan${a.n_challans === 1 ? '' : 's'}` : ' and their challans'}. It cannot be undone.</p>
        {err && <div className="alert err">{err}</div>}
        <div className="row" style={{ justifyContent: 'flex-end', gap: 8 }}><button className="btn" disabled={busy} onClick={() => setOpen(false)}>Cancel</button>
          <button className="btn primary" style={{ background: 'var(--red)', borderColor: 'var(--red)' }} disabled={busy} onClick={run}>{busy ? 'Deleting…' : 'Delete permanently'}</button></div>
      </div></div>}
  </>
}
