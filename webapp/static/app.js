/**
 * Traffic Violation System — Shared JS utilities.
 * Auth header injection, safe API helpers, token management, and UI toggles.
 */

const API_BASE = '';

function getToken() {
  return localStorage.getItem('tvs_token');
}

function setToken(token) {
  localStorage.setItem('tvs_token', token);
  document.cookie = `tvs_token=${token}; path=/; max-age=86400; SameSite=Lax`;
}

function clearToken() {
  localStorage.removeItem('tvs_token');
  localStorage.removeItem('tvs_email');
  document.cookie = 'tvs_token=; path=/; max-age=0; SameSite=Lax';
}

function getEmail() {
  return localStorage.getItem('tvs_email');
}

function setEmail(email) {
  localStorage.setItem('tvs_email', email);
}

function requireAuth() {
  if (!getToken()) {
    window.location.href = '/';
    return false;
  }
  return true;
}

function authHeaders() {
  const token = getToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function safeParseResponse(resp) {
  const text = await resp.text();
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text || resp.statusText || 'Unknown server response' };
  }
}

async function apiFetch(path, options = {}) {
  const headers = { ...authHeaders(), ...(options.headers || {}) };
  const resp = await fetch(API_BASE + path, { ...options, headers });

  if (resp.status === 401) {
    clearToken();
    window.location.href = '/';
    return null;
  }

  return resp;
}

async function apiJSON(path, options = {}) {
  const opts = { ...options };
  if (!opts.headers) opts.headers = {};
  if (!opts.headers['Content-Type'] && !(opts.body instanceof FormData)) {
    opts.headers['Content-Type'] = 'application/json';
  }
  const resp = await apiFetch(path, opts);
  if (!resp) return null;

  const data = await safeParseResponse(resp);
  if (!resp.ok) {
    const errorMsg = typeof data.detail === 'string' ? data.detail : (JSON.stringify(data.detail) || 'Request failed');
    throw new Error(errorMsg);
  }
  return data;
}

function logout() {
  clearToken();
  window.location.href = '/';
}

function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

function formatINR(val) {
  return '₹' + (Number(val) || 0).toLocaleString('en-IN');
}

function statusBadge(status) {
  const s = (status || 'unknown').toLowerCase();
  return `<span class="badge badge-${s}">${s.toUpperCase()}</span>`;
}

function severityBadge(level) {
  if (!level) return '';
  const lvl = level.toUpperCase();
  return `<span class="badge badge-${lvl.toLowerCase()}">${lvl}</span>`;
}

function setupNavbar() {
  const email = getEmail();
  const userEl = document.getElementById('nav-email');
  if (userEl && email) userEl.textContent = email;

  const logoutBtn = document.getElementById('btn-logout');
  if (logoutBtn) logoutBtn.addEventListener('click', logout);

  // Highlight active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.page-header nav a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}

const EYE_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>`;
const EYE_OFF_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>`;

function togglePassword(inputId, btn) {
  const input = document.getElementById(inputId);
  if (!input) return;

  if (input.type === 'password') {
    input.type = 'text';
    btn.innerHTML = EYE_OFF_SVG;
    btn.title = 'Hide password';
    btn.setAttribute('aria-label', 'Hide password');
  } else {
    input.type = 'password';
    btn.innerHTML = EYE_SVG;
    btn.title = 'Show password';
    btn.setAttribute('aria-label', 'Show password');
  }
}
