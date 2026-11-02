import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { Activity, BarChart3, ClipboardCheck, Cpu, FileText, Film, FolderOpen, LayoutDashboard, LogOut, Moon, Route, Settings, ShieldAlert, Sun, TrafficCone } from 'lucide-react'
import { useState } from 'react'
import { api } from './api'
import { useFetch } from './ui'
import { useMe } from './auth/AuthPages'

function Brand() {
  return <div className="brand">
    <div className="brand-mark"><svg width="12" height="18" viewBox="0 0 12 18"><circle cx="6" cy="3" r="2.4" fill="#ff6b6b" /><circle cx="6" cy="9" r="2.4" fill="#ffc857" /><circle cx="6" cy="15" r="2.4" fill="#6ee7b7" /></svg></div>
    <div>TraffiSense<small>Traffic management &amp; enforcement</small></div>
  </div>
}

export default function Layout() {
  const me = useMe()
  const nav = useNavigate()
  const { data: st } = useFetch(() => api.get('/api/stats'), [], 15000)
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || '')
  const toggle = () => {
    const dark = theme === 'dark'
    const t = dark ? 'light' : 'dark'
    document.documentElement.dataset.theme = t; setTheme(t)
    try { localStorage.setItem('tvp-theme', t) } catch { /* ignore */ }
  }
  const logout = async () => { await api.post('/api/auth/logout'); nav('/login') }
  const L = (to: string, icon: any, label: string, count?: number) =>
    <NavLink to={to} end={to === '/'} className={({ isActive }) => 'nav' + (isActive ? ' active' : '')}>{icon}{label}{!!count && <span className="count">{count}</span>}</NavLink>
  return <div className="shell">
    <aside className="side" aria-label="Main navigation">
      <Brand />
      {L('/', <LayoutDashboard size={16} />, 'Dashboard')}
      {L('/analyses', <Film size={16} />, 'Video analyses', st?.processing)}
      <div className="nav-group">Enforcement</div>
      {L('/violations', <ShieldAlert size={16} />, 'Violations')}
      {L('/review', <ClipboardCheck size={16} />, 'Human review', st?.pending_review)}
      {L('/challans', <FileText size={16} />, 'Challans')}
      <div className="nav-group">Platform</div>
      {L('/system', <Cpu size={16} />, 'Models & system')}
      {L('/evidence', <FolderOpen size={16} />, 'Evidence')}
      {L('/twin', <Activity size={16} />, 'Digital twin')}
      {L('/lanes', <Route size={16} />, 'Lane detection')}
      {/* the Unity simulator is a separate full-window page served at /sim3d/, so it opens in its own tab */}
      <a className="nav" href="/sim3d/" target="_blank" rel="noopener"><TrafficCone size={16} />3D Simulator</a>
      {L('/analytics', <BarChart3 size={16} />, 'Analytics')}
      {L('/settings', <Settings size={16} />, 'Settings')}
      <div className="side-foot">
        <div style={{ minWidth: 0, flex: 1 }}><div className="small" style={{ fontWeight: 500 }}>{me?.name}</div><div className="tiny faint" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{me?.email || me?.phone}</div></div>
        <button className="btn ghost sm" onClick={toggle} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}</button>
        <button className="btn ghost sm" onClick={logout} aria-label="Sign out"><LogOut size={15} /></button>
      </div>
    </aside>
    <main className="main"><Outlet /></main>
  </div>
}
