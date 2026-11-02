import { createContext, useContext, useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { CheckCircle2, Eye, EyeOff, FileCheck2, RefreshCw, ScanEye, UserCheck } from 'lucide-react'
import { api, ApiError } from '../api'
import { Spinner } from '../ui'

type Me = { id: string; name: string; email?: string; phone?: string; role: string }
const MeCtx = createContext<Me | null>(null)
export const useMe = () => useContext(MeCtx)

export function AuthGate({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Me | null | undefined>(undefined)
  const loc = useLocation()
  useEffect(() => { api.get<Me>('/api/auth/me').then(setMe).catch(() => setMe(null)) }, [loc.pathname === '/'])
  if (me === undefined) return <div style={{ display: 'grid', placeItems: 'center', height: '100%' }}><Spinner /></div>
  if (me === null) return <Navigate to="/login" state={{ from: loc.pathname }} replace />
  return <MeCtx.Provider value={me}>{children}</MeCtx.Provider>
}

function Shell({ children }: { children: ReactNode }) {
  return <div className="auth">
    <div className="auth-side">
      <div className="brand" style={{ color: '#0f1f38' }}>
        <div className="brand-mark"><svg width="12" height="18" viewBox="0 0 12 18"><circle cx="6" cy="3" r="2.4" fill="#ff6b6b" /><circle cx="6" cy="9" r="2.4" fill="#ffc857" /><circle cx="6" cy="15" r="2.4" fill="#6ee7b7" /></svg></div>
        <div>TraffiSense<small style={{ color: '#4a5b75' }}>Traffic management &amp; enforcement</small></div>
      </div>
      <h2>Evidence-first traffic enforcement from recorded video.</h2>
      <p>Every challan is backed by tracked, multi-frame evidence and a plain-language explanation. Uncertain cases go to a person.</p>
      <ul>
        <li><ScanEye size={16} /> Multi-frame detection and tracking</li>
        <li><UserCheck size={16} /> Human review for uncertain cases</li>
        <li><FileCheck2 size={16} /> Downloadable challans with evidence</li>
      </ul>
    </div>
    <div className="auth-form"><div className="auth-box fade-in">{children}</div></div>
  </div>
}

function DevNotice({ code }: { code?: string }) {
  const [cfg, setCfg] = useState<any>(null)
  useEffect(() => { api.get('/api/auth/config').then(setCfg).catch(() => {}) }, [])
  if (!cfg?.dev_mode) return null
  return <div className="devbox"><b>Developer mode</b> — no email/SMS provider is configured, so codes are not sent.
    {code ? <> Your code: <span className="mono" style={{ fontWeight: 600 }}>{code}</span></> : ' Codes appear here and in app/data/dev_outbox.log.'}</div>
}

function Captcha({ onChange, nonce }: { onChange: (tok: string, ans: string) => void; nonce: number }) {
  const [c, setC] = useState<{ image: string; token: string } | null>(null)
  const [ans, setAns] = useState('')
  const load = () => { setAns(''); api.get('/api/auth/captcha').then(d => { setC(d); onChange(d.token, '') }) }
  useEffect(load, [nonce])
  return <div className="field">
    <label htmlFor="cap">Type the characters</label>
    <div className="captcha">
      {c ? <img src={c.image} alt="CAPTCHA characters" /> : <div className="skeleton" style={{ width: 180, height: 44 }} />}
      <button type="button" className="btn ghost sm" onClick={load} aria-label="New image"><RefreshCw size={14} /></button>
    </div>
    <input id="cap" className="input" autoComplete="off" value={ans} maxLength={6}
      onChange={e => { setAns(e.target.value); c && onChange(c.token, e.target.value) }} />
  </div>
}

function Pw({ value, onChange, meter, label = 'Password', id = 'pw' }: { value: string; onChange: (v: string) => void; meter?: boolean; label?: string; id?: string }) {
  const [show, setShow] = useState(false)
  const rules = [value.length >= 10, /[a-z]/.test(value) && /[A-Z]/.test(value), /\d/.test(value), /[^\w\s]/.test(value)]
  return <div className="field">
    <label htmlFor={id}>{label}</label>
    <div style={{ position: 'relative' }}>
      <input id={id} className="input" type={show ? 'text' : 'password'} value={value} onChange={e => onChange(e.target.value)} autoComplete={meter ? 'new-password' : 'current-password'} />
      <button type="button" className="btn ghost sm" style={{ position: 'absolute', right: 3, top: 3 }} onClick={() => setShow(!show)} aria-label={show ? 'Hide password' : 'Show password'}>{show ? <EyeOff size={14} /> : <Eye size={14} />}</button>
    </div>
    {meter && <><div className="pwmeter">{rules.map((r, i) => <i key={i} className={r ? 'on' : ''} />)}</div>
      <div className="hint">10+ characters, upper and lower case, a number and a symbol</div></>}
  </div>
}

export function Login() {
  const nav = useNavigate(); const loc = useLocation() as any
  const [id, setId] = useState(''); const [pw, setPw] = useState('')
  const [cap, setCap] = useState<[string, string] | null>(null); const [needCap, setNeedCap] = useState(false); const [nonce, setNonce] = useState(0)
  const [err, setErr] = useState(''); const [busy, setBusy] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setErr(''); setBusy(true)
    try {
      const r = await api.post('/api/auth/login', { identifier: id, password: pw, captcha_token: cap?.[0], captcha: cap?.[1] })
      if (r.needs_verification) nav('/verify', { state: { identifier: id, dev_code: r.dev_code } })
      else nav(loc.state?.from || '/')
    } catch (e: any) {
      setErr(e.message)
      if (e instanceof ApiError && e.data?.captcha_required) { setNeedCap(true); setNonce(n => n + 1) }
    } finally { setBusy(false) }
  }
  return <Shell>
    <h1>Sign in</h1><p className="sub">Use your email address or phone number.</p>
    {err && <div className="alert err" role="alert">{err}</div>}
    <form onSubmit={submit}>
      <div className="field"><label htmlFor="id">Email or phone</label><input id="id" className="input" value={id} onChange={e => setId(e.target.value)} autoComplete="username" autoFocus /></div>
      <Pw value={pw} onChange={setPw} />
      {needCap && <Captcha nonce={nonce} onChange={(t, a) => setCap([t, a])} />}
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}><span /><Link to="/forgot" className="small">Forgot password?</Link></div>
      <button className="btn primary block" disabled={busy || !id || !pw}>{busy ? <Spinner /> : 'Sign in'}</button>
    </form>
    <p className="small muted" style={{ marginTop: 20 }}>No account? <Link to="/signup">Create one</Link></p>
  </Shell>
}

export function Signup() {
  const nav = useNavigate()
  const [f, setF] = useState({ name: '', identifier: '', password: '', confirm: '' })
  const [cap, setCap] = useState<[string, string]>(['', '']); const [nonce, setNonce] = useState(0)
  const [err, setErr] = useState(''); const [busy, setBusy] = useState(false)
  const set = (k: string) => (e: any) => setF({ ...f, [k]: e.target.value })
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setErr('')
    if (f.password !== f.confirm) return setErr('Passwords do not match')
    setBusy(true)
    try {
      const r = await api.post('/api/auth/signup', { name: f.name, identifier: f.identifier, password: f.password, captcha_token: cap[0], captcha: cap[1] })
      nav('/verify', { state: { identifier: f.identifier, dev_code: r.dev_code } })
    } catch (e: any) { setErr(e.message); setNonce(n => n + 1) } finally { setBusy(false) }
  }
  return <Shell>
    <h1>Create account</h1><p className="sub">We'll send a verification code to confirm it's you.</p>
    {err && <div className="alert err" role="alert">{err}</div>}
    <form onSubmit={submit}>
      <div className="field"><label htmlFor="n">Full name</label><input id="n" className="input" value={f.name} onChange={set('name')} autoComplete="name" autoFocus /></div>
      <div className="field"><label htmlFor="i">Email or phone</label><input id="i" className="input" value={f.identifier} onChange={set('identifier')} autoComplete="username" /></div>
      <Pw value={f.password} onChange={v => setF({ ...f, password: v })} meter />
      <Pw value={f.confirm} onChange={v => setF({ ...f, confirm: v })} label="Confirm password" id="pw2" />
      <Captcha nonce={nonce} onChange={(t, a) => setCap([t, a])} />
      <button className="btn primary block" disabled={busy}>{busy ? <Spinner /> : 'Create account'}</button>
    </form>
    <p className="small muted" style={{ marginTop: 20 }}>Already registered? <Link to="/login">Sign in</Link></p>
  </Shell>
}

function Otp({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const refs = useRef<(HTMLInputElement | null)[]>([])
  const put = (i: number, ch: string) => {
    const d = ch.replace(/\D/g, '')
    if (d.length > 1) { onChange(d.slice(0, 6)); refs.current[Math.min(5, d.length)]?.focus(); return }
    const a = value.padEnd(6, ' ').split(''); a[i] = d || ' '; onChange(a.join('').trimEnd())
    if (d && i < 5) refs.current[i + 1]?.focus()
  }
  return <div className="otp" role="group" aria-label="Verification code">
    {Array.from({ length: 6 }, (_, i) => <input key={i} ref={el => { refs.current[i] = el }} className="input" inputMode="numeric" maxLength={6}
      value={value[i]?.trim() || ''} onChange={e => put(i, e.target.value)} aria-label={`Digit ${i + 1}`}
      onKeyDown={e => { if (e.key === 'Backspace' && !value[i]?.trim() && i) refs.current[i - 1]?.focus() }} autoFocus={i === 0} />)}
  </div>
}

export function Verify() {
  const nav = useNavigate(); const loc = useLocation() as any
  const ident = loc.state?.identifier
  const [code, setCode] = useState(''); const [dev, setDev] = useState(loc.state?.dev_code)
  const [err, setErr] = useState(''); const [msg, setMsg] = useState(''); const [busy, setBusy] = useState(false)
  if (!ident) return <Navigate to="/login" replace />
  const submit = async (e?: FormEvent) => {
    e?.preventDefault(); setErr(''); setBusy(true)
    try { await api.post('/api/auth/verify', { identifier: ident, code }); nav('/') } catch (e: any) { setErr(e.message) } finally { setBusy(false) }
  }
  const resend = async () => {
    setErr(''); try { const r = await api.post('/api/auth/resend', { identifier: ident }); setDev(r.dev_code); setMsg('A new code was sent.') } catch (e: any) { setErr(e.message) }
  }
  return <Shell>
    <h1>Verify your account</h1><p className="sub">Enter the 6-digit code sent to <b>{ident}</b>.</p>
    <DevNotice code={dev} />
    {err && <div className="alert err" role="alert">{err}</div>}
    {msg && <div className="alert ok"><CheckCircle2 size={14} /> {msg}</div>}
    <form onSubmit={submit}>
      <div className="field"><Otp value={code} onChange={setCode} /></div>
      <button className="btn primary block" disabled={busy || code.replace(/\s/g, '').length !== 6}>{busy ? <Spinner /> : 'Verify and continue'}</button>
    </form>
    <p className="small muted" style={{ marginTop: 20 }}>Didn't get it? <button className="btn ghost sm" onClick={resend}>Send a new code</button></p>
  </Shell>
}

export function Forgot() {
  const nav = useNavigate()
  const [step, setStep] = useState<1 | 2>(1)
  const [id, setId] = useState(''); const [code, setCode] = useState(''); const [pw, setPw] = useState('')
  const [cap, setCap] = useState<[string, string]>(['', '']); const [nonce, setNonce] = useState(0)
  const [dev, setDev] = useState<string>(); const [err, setErr] = useState(''); const [busy, setBusy] = useState(false)
  const send = async (e: FormEvent) => {
    e.preventDefault(); setErr(''); setBusy(true)
    try { const r = await api.post('/api/auth/forgot', { identifier: id, captcha_token: cap[0], captcha: cap[1] }); setDev(r.dev_code); setStep(2) }
    catch (e: any) { setErr(e.message); setNonce(n => n + 1) } finally { setBusy(false) }
  }
  const reset = async (e: FormEvent) => {
    e.preventDefault(); setErr(''); setBusy(true)
    try { await api.post('/api/auth/reset', { identifier: id, code, password: pw }); nav('/login', { state: {} }) }
    catch (e: any) { setErr(e.message) } finally { setBusy(false) }
  }
  return <Shell>
    <h1>Reset password</h1>
    <p className="sub">{step === 1 ? 'We will send a code to your email or phone.' : `If ${id} has an account, a code was sent to it.`}</p>
    {step === 2 && <DevNotice code={dev} />}
    {err && <div className="alert err" role="alert">{err}</div>}
    {step === 1 ? <form onSubmit={send}>
      <div className="field"><label htmlFor="fi">Email or phone</label><input id="fi" className="input" value={id} onChange={e => setId(e.target.value)} autoFocus /></div>
      <Captcha nonce={nonce} onChange={(t, a) => setCap([t, a])} />
      <button className="btn primary block" disabled={busy || !id}>{busy ? <Spinner /> : 'Send code'}</button>
    </form> : <form onSubmit={reset}>
      <div className="field"><label>Code</label><Otp value={code} onChange={setCode} /></div>
      <Pw value={pw} onChange={setPw} meter label="New password" />
      <button className="btn primary block" disabled={busy}>{busy ? <Spinner /> : 'Set new password'}</button>
    </form>}
    <p className="small muted" style={{ marginTop: 20 }}><Link to="/login">Back to sign in</Link></p>
  </Shell>
}
