import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import './styles.css'
import { AuthGate, Login, Signup, Verify, Forgot } from './auth/AuthPages'
import Layout from './Layout'
import Dashboard from './pages/Dashboard'
import AnalysisPage from './pages/Analysis'
import Analyses from './pages/Analyses'
import Violations from './pages/Violations'
import Review from './pages/Review'
import Challans, { ChallanDetail } from './pages/Challans'
import System from './pages/System'
import TwinPage from './pages/Twin'
import SignalSimPage from './pages/SignalSim'
import Lanes from './pages/Lanes'
import Evidence from './pages/Evidence'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'

try { const t = localStorage.getItem('tvp-theme'); if (t) document.documentElement.dataset.theme = t } catch { /* private mode */ }

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/verify" element={<Verify />} />
        <Route path="/forgot" element={<Forgot />} />
        <Route element={<AuthGate><Layout /></AuthGate>}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyses" element={<Analyses />} />
          <Route path="/analyses/:id" element={<AnalysisPage />} />
          <Route path="/violations" element={<Violations />} />
          <Route path="/review" element={<Review />} />
          <Route path="/review/:id" element={<Review />} />
          <Route path="/challans" element={<Challans />} />
          <Route path="/challans/:id" element={<ChallanDetail />} />
          <Route path="/system" element={<System />} />
          <Route path="/signal/:id" element={<SignalSimPage />} />
          <Route path="/twin" element={<TwinPage />} />
          <Route path="/twin/:id" element={<TwinPage />} />
          <Route path="/lanes" element={<Lanes />} />
          <Route path="/lanes/:id" element={<Lanes />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
