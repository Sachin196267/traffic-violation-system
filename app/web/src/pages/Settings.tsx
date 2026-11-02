import { useState, type FormEvent } from 'react'
import { api } from '../api'
import { Card, PageHead, useFetch } from '../ui'
import { useMe } from '../auth/AuthPages'

export const getDefaults = () => { try { return JSON.parse(localStorage.getItem('tvp-defaults') || '{}') } catch { return {} } }

export default function Settings() {
  const me = useMe()
  const { data: cfg } = useFetch<any>(() => api.get('/api/auth/config'), [])
  const [name, setName] = useState(me?.name || ''); const [msg, setMsg] = useState<[string, string] | null>(null)
  const [pw, setPw] = useState({ current: '', password: '' })
  const [def, setDef] = useState(() => ({ checks: ['helmet', 'triple'], max_sec: '120', ...getDefaults() }))
  const save = async (e: FormEvent, fn: () => Promise<any>, ok: string) => { e.preventDefault(); try { await fn(); setMsg(['ok', ok]) } catch (x: any) { setMsg(['err', x.message]) } }
  const theme = (t: string) => { document.documentElement.dataset.theme = t; try { localStorage.setItem('tvp-theme', t) } catch { /* */ } }
  const saveDef = (d: any) => { setDef(d); try { localStorage.setItem('tvp-defaults', JSON.stringify(d)) } catch { /* */ } }
  return <div className="page" style={{ maxWidth: 900 }}>
    <PageHead title="Settings" />
    {msg && <div className={`alert ${msg[0]}`}>{msg[1]}</div>}
    <div className="grid">
      <Card title="Profile"><form onSubmit={e => save(e, () => api.post('/api/me', { name }), 'Profile saved')} className="row" style={{ alignItems: 'flex-end' }}>
        <div className="field" style={{ flex: 1, marginBottom: 0 }}><label htmlFor="nm">Name</label><input id="nm" className="input" value={name} onChange={e => setName(e.target.value)} /></div>
        <button className="btn primary">Save</button></form>
        <div className="small muted" style={{ marginTop: 10 }}>Signed in as {me?.email || me?.phone} · role: {me?.role}</div></Card>
      <Card title="Password"><form onSubmit={e => save(e, () => api.post('/api/me/password', pw).then(() => setPw({ current: '', password: '' })), 'Password changed; other sessions were signed out')}>
        <div className="grid g2"><div className="field"><label htmlFor="c">Current password</label><input id="c" type="password" className="input" value={pw.current} onChange={e => setPw({ ...pw, current: e.target.value })} autoComplete="current-password" /></div>
          <div className="field"><label htmlFor="n">New password</label><input id="n" type="password" className="input" value={pw.password} onChange={e => setPw({ ...pw, password: e.target.value })} autoComplete="new-password" /></div></div>
        <button className="btn">Change password</button></form></Card>
      <Card title="Appearance"><div className="seg"><button onClick={() => theme('light')}>Light</button><button onClick={() => theme('dark')}>Dark</button></div></Card>
      <Card title="Upload defaults">
        <div className="row wrap gap16"><span className="small muted">All checks run on every video.</span>
          <label className="row small">Analyse the first <select value={def.max_sec} onChange={e => saveDef({ ...def, max_sec: e.target.value })} style={{ width: 130 }}>
            <option value="30">30 seconds</option><option value="60">1 minute</option><option value="120">2 minutes</option><option value="300">5 minutes</option><option value="0">whole video</option></select></label></div>
      </Card>
      <Card title="Delivery of verification codes"><p className="small muted" style={{ margin: 0 }}>{cfg?.dev_mode
        ? 'Developer mode: no email/SMS provider configured. Set TVP_SMTP_HOST, TVP_SMTP_USER, TVP_SMTP_PASS and TVP_SMTP_FROM on the server to send real emails.'
        : `Codes are sent by ${cfg?.delivery}.`}</p></Card>
    </div>
  </div>
}
