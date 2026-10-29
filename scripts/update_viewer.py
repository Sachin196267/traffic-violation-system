import re

with open('viewer_3d.html', 'r', encoding='utf-8') as f:
    content = f.read()

# CSS to add
css_to_add = """
    /* SLIDE-OUT E-CHALLAN SLIP DRAWER */
    #challan-drawer {
      position: fixed;
      top: 14px;
      right: 18px;
      bottom: 14px;
      width: 440px;
      z-index: 100;
      background: rgba(13, 20, 36, 0.95);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 14px;
      box-shadow: -12px 18px 50px rgba(0, 0, 0, 0.8), 0 0 25px rgba(239, 68, 68, 0.25);
      transform: translateX(120%);
      transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #challan-drawer.open {
      transform: translateX(0);
    }
    .slip-header-banner {
      background: linear-gradient(135deg, #1e3a8a, #0f172a);
      border-bottom: 2px solid #2563eb;
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .slip-emblem-seal {
      width: 32px;
      height: 32px;
      background: #3b82f6;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      box-shadow: 0 0 12px rgba(59, 130, 246, 0.5);
    }
    .slip-body-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .slip-fine-card {
      background: linear-gradient(135deg, rgba(220, 38, 38, 0.25), rgba(185, 28, 28, 0.12));
      border: 1px solid rgba(239, 68, 68, 0.4);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .slip-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .slip-table th {
      background: #1e293b;
      color: #94a3b8;
      text-align: left;
      padding: 6px 10px;
      font-weight: 600;
      width: 40%;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .slip-table td {
      padding: 6px 10px;
      color: #f1f5f9;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-family: monospace;
      font-size: 12px;
    }
    .slip-evidence-img-container {
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      overflow: hidden;
      background: #0b0f19;
    }
    .slip-evidence-img-container img {
      width: 100%;
      display: block;
      max-height: 180px;
      object-fit: cover;
    }
    .slip-btn-dl {
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      color: #ffffff;
      border: none;
      padding: 9px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      text-decoration: none;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
      transition: all 0.2s;
    }
    .slip-btn-dl:hover {
      background: #1d4ed8;
      transform: translateY(-1px);
    }
    .close-slip-btn {
      background: rgba(255, 255, 255, 0.1);
      border: none;
      color: #94a3b8;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
    }
    .close-slip-btn:hover {
      background: rgba(239, 68, 68, 0.3);
      color: #fff;
    }
"""

if '#challan-drawer' not in content:
    content = content.replace('</style>', css_to_add + '\n  </style>')

drawer_html = """
  <!-- SLIDE-OUT E-CHALLAN SLIP DRAWER (GENERATES AUTOMATICALLY ON VIOLATION) -->
  <div id="challan-drawer">
    <div class="slip-header-banner">
      <div style="display: flex; gap: 10px; align-items: center;">
        <div class="slip-emblem-seal">🏛️</div>
        <div>
          <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.5px; color: #f8fafc;">GOVERNMENT OF MAHARASHTRA</div>
          <div style="font-size: 10px; color: #93c5fd;">INTELLIGENT TRAFFIC ENFORCEMENT BRANCH</div>
          <div style="font-size: 9.5px; color: #60a5fa; font-weight: 700; letter-spacing: 0.8px;">AUTOMATED E-CHALLAN EVIDENCE SLIP</div>
        </div>
      </div>
      <button class="close-slip-btn" onclick="closeChallanDrawer()">✕</button>
    </div>

    <div class="slip-body-scroll">
      <!-- Fine Callout Banner -->
      <div class="slip-fine-card">
        <div>
          <div style="font-size: 9px; text-transform: uppercase; letter-spacing: 0.5px; color: #fca5a5;">Total Statutory Penalty</div>
          <div style="font-size: 19px; font-weight: 800; color: #ffffff;" id="drawer-fine">INR 5,000</div>
        </div>
        <div style="background: #b91c1c; color: #fff; font-size: 9.5px; font-weight: 700; padding: 4px 8px; border-radius: 4px; letter-spacing: 0.5px;" id="drawer-badge-status">RED LIGHT BREACH</div>
      </div>

      <!-- Metadata Table -->
      <table class="slip-table">
        <tr><th>Challan ID</th><td id="drawer-id">TVS-20260919-000120-0042</td></tr>
        <tr><th>Issued At</th><td id="drawer-time">2026-09-19 14:15:22</td></tr>
        <tr><th>Offense Type</th><td style="color: #ef4444; font-weight: 700;" id="drawer-offense">RED_LIGHT_VIOLATION</td></tr>
        <tr><th>Offending Vehicle</th><td id="drawer-vehicle">Vehicle #V26 (Red Sedan)</td></tr>
        <tr><th>Lane / Approach</th><td id="drawer-lane">Lane S1 (Southbound Approach)</td></tr>
        <tr><th>Speed Recorded</th><td id="drawer-speed">38.4 km/h</td></tr>
        <tr><th>Intersection</th><td>Sector 20, Kharghar, Navi Mumbai</td></tr>
      </table>

      <!-- Evidence Image Preview -->
      <div style="font-size: 10px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;">EVIDENCE SNAPSHOT (CONTEXT & DETAIL ZOOM)</div>
      <div class="slip-evidence-img-container">
        <img id="drawer-evidence-img" src="outputs/test_challan_output/evidence_TVS-20260919-000120-0042.jpg" alt="Evidence Composite" />
      </div>

      <!-- Explainable AI Section -->
      <div style="font-size: 10px; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;">EXPLAINABLE AI (XAI) EVIDENCE LOG</div>
      <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 8px 10px; font-size: 11px;">
        <div style="color: #38bdf8; font-weight: 600; margin-bottom: 2px;">Decision Score: 0.98 | Threshold: 0.50</div>
        <div style="color: #cbd5e1;">&bull; <b>Signal Phase:</b> Stop line crossed 4.2s after Red Signal initiation.</div>
        <div style="color: #cbd5e1;">&bull; <b>Spatial Encroachment:</b> Contact point entered pedestrian zebra crossing zone.</div>
      </div>

      <!-- Actions -->
      <div style="display: flex; gap: 8px; margin-top: 4px;">
        <a id="drawer-pdf-link" href="outputs/test_challan_output/TVS-20260919-000120-0042.pdf" target="_blank" class="slip-btn-dl" style="flex: 1;">
          📄 Download PDF Slip
        </a>
        <button class="btn" style="padding: 9px 12px; font-size: 11px;" onclick="endorseChallan()">
          ✅ Endorse
        </button>
      </div>

      <div style="font-size: 9.5px; color: #64748b; line-height: 1.3; text-align: justify; margin-top: 2px;">
        Statutory Notice: Automatically issued by Intelligent Traffic Management System. Review source video before legal penalty dispatch.
      </div>
    </div>
  </div>
"""

if 'id="challan-drawer"' not in content:
    content = content.replace('</body>', drawer_html + '\n</body>')

js_drawer = """
  // --- 15. AUTOMATIC SIDE E-CHALLAN SLIP GENERATION ---
  const challanDrawer = document.getElementById('challan-drawer');
  let lastSlipTriggerTime = 0;

  window.openChallanSlip = function(data) {
    if (!data) data = {
      id: 'TVS-20260919-000120-0042',
      fine: 'INR 5,000',
      offense: 'RED_LIGHT_VIOLATION',
      vehicle: 'Vehicle #V26 (Red Sedan)',
      lane: 'Lane S1 (Southbound Approach)',
      speed: '38.4 km/h',
      status: 'RED LIGHT BREACH'
    };

    document.getElementById('drawer-id').innerText = data.id || 'TVS-20260919-000120-0042';
    document.getElementById('drawer-fine').innerText = data.fine || 'INR 5,000';
    document.getElementById('drawer-offense').innerText = data.offense || 'RED_LIGHT_VIOLATION';
    document.getElementById('drawer-vehicle').innerText = data.vehicle || 'Vehicle #V26 (Red Sedan)';
    document.getElementById('drawer-lane').innerText = data.lane || 'Lane S1 (Southbound)';
    document.getElementById('drawer-speed').innerText = data.speed || '38.4 km/h';
    document.getElementById('drawer-badge-status').innerText = data.status || 'VIOLATION DETECTED';
    document.getElementById('drawer-time').innerText = new Date().toISOString().replace('T', ' ').substring(0, 19);

    if (challanDrawer) {
      challanDrawer.classList.add('open');
    }
  };

  window.closeChallanDrawer = function() {
    if (challanDrawer) {
      challanDrawer.classList.remove('open');
    }
  };

  window.endorseChallan = function() {
    alert('✅ Challan TVS-20260919-000120-0042 verified and endorsed by reviewing authority.');
    closeChallanDrawer();
  };

  if (document.getElementById('violations-card')) {
    document.getElementById('violations-card').style.cursor = 'pointer';
    document.getElementById('violations-card').title = 'Click to open generated e-Challan slip';
    document.getElementById('violations-card').addEventListener('click', () => {
      openChallanSlip();
    });
  }
"""

if 'openChallanSlip' not in content:
    content = content.replace('// --- 13. ANIMATION LOOP ---', js_drawer + '\n  // --- 13. ANIMATION LOOP ---')

trigger_code = """
      // Auto-trigger side challan slip as soon as red light breach occurs!
      if (violatorCar.position.z < 22 && violatorCar.position.z > 14) {
        const nowMs = Date.now();
        if (nowMs - lastSlipTriggerTime > 12000) {
          lastSlipTriggerTime = nowMs;
          openChallanSlip({
            id: 'TVS-20260919-000120-0042',
            fine: 'INR 5,000',
            offense: 'RED_LIGHT_VIOLATION',
            vehicle: 'Vehicle #V26 (Red Sedan)',
            lane: 'Lane S1 (Southbound Approach)',
            speed: '38.4 km/h',
            status: 'RED LIGHT BREACH'
          });
        }
      }
"""

if 'lastSlipTriggerTime' not in content and 'violatorCar.position.z -= violatorCar.position.z > -ARM_LEN ? 0.35 : -ARM_LEN;' in content:
    content = content.replace(
        'violatorCar.position.z -= violatorCar.position.z > -ARM_LEN ? 0.35 : -ARM_LEN;',
        trigger_code + '\n      violatorCar.position.z -= violatorCar.position.z > -ARM_LEN ? 0.35 : -ARM_LEN;'
    )

with open('viewer_3d.html', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated viewer_3d.html successfully with Side Challan Drawer!")
