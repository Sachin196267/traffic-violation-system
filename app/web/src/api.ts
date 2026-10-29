// Thin client for the FastAPI backend. Cookies carry the session; the X-TVP header is required on writes.
export class ApiError extends Error {
  status: number; data: any
  constructor(status: number, message: string, data?: any) { super(message); this.status = status; this.data = data }
}

async function req<T>(method: string, path: string, body?: any): Promise<T> {
  const init: RequestInit = { method, credentials: 'same-origin', headers: { 'X-TVP': '1' } }
  if (body instanceof FormData) init.body = body
  else if (body !== undefined) { init.body = JSON.stringify(body); (init.headers as any)['Content-Type'] = 'application/json' }
  const r = await fetch(path, init)
  const txt = await r.text()
  const data = txt ? JSON.parse(txt) : null
  if (!r.ok) {
    const d = data?.detail
    const msg = typeof d === 'string' ? d : d?.message || (Array.isArray(d) ? 'Please check the form' : r.statusText)
    throw new ApiError(r.status, msg, d)
  }
  return data as T
}

export const api = {
  get: <T = any>(p: string) => req<T>('GET', p),
  post: <T = any>(p: string, b?: any) => req<T>('POST', p, b ?? {}),
  del: <T = any>(p: string) => req<T>('DELETE', p),
  file: (aid: string, rel: string) => `/api/analyses/${aid}/files/${rel}`,
}

export type Offence = { id: string; violation: string; confidence: number; certain: boolean; t: number; analysis_id: string }
export type Linked = { id: string; violation: string; status: string; t: number; analysis_id: string; filename: string }
export type Finding = {
  id: string; analysis_id: string; detector: string; violation: string; track: string; t: number; t_start: number; t_end: number
  confidence: number; model_decision: string; route: 'confirmed' | 'review' | 'exempt'
  status: 'confirmed' | 'pending_review' | 'needs_evidence' | 'rejected' | 'challan' | 'exempt' | 'duplicate'
  box: number[] | null; reasons: string[]; uncertain_reasons: string[]; evidence: any; emergency: string | null
  vehicle: string; model_info: any; created: number; filename?: string; reviews?: any[]; challan?: any
  offences?: Offence[] | null; duplicate_of?: string | null; merged?: Finding[]; merged_into?: Linked | null
  why?: string   // merged detections: why it belongs to the case
}
/** every offence of the vehicle in this one case (one challan or review per vehicle) */
export const offenceText = (f: { violation: string; offences?: Offence[] | null }) => [f.violation, ...(f.offences || []).map(o => o.violation)].join(' + ')
export type Analysis = {
  id: string; filename: string; created: number; status: string; stage: string; progress: number; error: string | null
  options: any; meta: any; finished: number | null; findings?: Finding[]; challans?: any[]
  n_findings?: number; n_review?: number; n_challans?: number
}

export const fmtTime = (s: number) => { const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(1).padStart(4, '0')}` }
export const fmtDate = (t: number) => new Date(t * 1000).toLocaleString(undefined, { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
export const ago = (t: number) => {
  const s = Date.now() / 1000 - t
  if (s < 60) return 'just now'; if (s < 3600) return `${Math.floor(s / 60)} min ago`; if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return fmtDate(t)
}
