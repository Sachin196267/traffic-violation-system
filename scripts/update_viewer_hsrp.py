with open('viewer_3d.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the fine callout in drawer with HSRP badge + VAHAN registry
old_drawer_target = '<!-- Fine Callout Banner -->'
hsrp_block = """<!-- Indian High-Security Registration Plate (HSRP) Badge -->
      <div style="background: #ffffff; color: #111827; border-radius: 8px; border: 2.5px solid #1f2937; padding: 6px 12px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 14px rgba(0,0,0,0.5);">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="background: #1e3a8a; color: #ffffff; padding: 3px 6px; border-radius: 4px; display: flex; flex-direction: column; align-items: center; font-size: 8px; font-weight: 800; line-height: 1.1;">
            <span style="font-size: 10px;">🇮🇳</span>
            <span>IND</span>
          </div>
          <div style="font-family: 'Courier New', monospace, sans-serif; font-size: 20px; font-weight: 900; letter-spacing: 2px; color: #111827;" id="drawer-plate-num">
            MH 46 AX 4296
          </div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 8.5px; font-weight: 800; color: #15803d; background: #dcfce7; padding: 2px 7px; border-radius: 4px; display: inline-block;">ANPR: 98.4%</div>
          <div style="font-size: 7.5px; color: #6b7280; margin-top: 2px; font-weight: 600;">HSRP HOLOGRAM VALID</div>
        </div>
      </div>

      <!-- VAHAN Vehicle Registry Info -->
      <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 8px 12px; font-size: 11px;">
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px; margin-bottom: 5px;">
          <span style="color: #94a3b8;">Registered Owner:</span>
          <b style="color: #f1f5f9;">Rajesh M. Sharma</b>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
          <span style="color: #94a3b8;">Vehicle Model:</span>
          <span style="color: #e2e8f0;">Honda City ZX (Radiant Red)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
          <span style="color: #94a3b8;">RTO Registered:</span>
          <span style="color: #38bdf8; font-weight: 600;">MH-46 Navi Mumbai (Panvel)</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #94a3b8;">Insurance & PUC:</span>
          <span style="color: #34d399; font-weight: 600;">Certified Valid (Dec 2026)</span>
        </div>
      </div>

      <!-- Fine Callout Banner -->"""

if 'drawer-plate-num' not in content:
    content = content.replace('<!-- Fine Callout Banner -->', hsrp_block)

# Update Selected Vehicle card to show Number Plate
old_sel_card = '<div class="info-row"><span>Type:</span><span class="info-val" id="sel-type">Sedan Car</span></div>'
new_sel_card = """<div class="info-row"><span>Number Plate:</span><span class="info-val" style="color: #facc15; font-family: monospace; font-weight: 700;" id="sel-plate">MH 46 AX 4296</span></div>
    <div class="info-row"><span>Type:</span><span class="info-val" id="sel-type">Sedan Car</span></div>"""

if 'id="sel-plate"' not in content:
    content = content.replace(old_sel_card, new_sel_card)

# Update selectVehicle function to set plate
old_select_func = "document.getElementById('sel-type').innerText = v.type;"
new_select_func = """document.getElementById('sel-type').innerText = v.type;
      const plateEl = document.getElementById('sel-plate');
      if (plateEl) {
        const plates = { 'V26': 'MH 46 AX 4296', 'V03': 'MH 46 EM 1080', 'V13': 'MH 01 BR 4512', 'V12': 'MH 43 AT 8891', 'V72': 'MH 46 CZ 2045', 'V20': 'MH 46 R 6721' };
        plateEl.innerText = plates[v.id] || 'MH 46 AX 4296';
      }"""
if 'plates[v.id]' not in content:
    content = content.replace(old_select_func, new_select_func)

with open('viewer_3d.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("HSRP Indian Number Plate added to viewer_3d.html successfully!")
