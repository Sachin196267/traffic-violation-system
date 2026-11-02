using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// Runs SimCore, mirrors it in 3D (WorldBuilder, VehicleModels), drives the cameras (CameraRig) and draws the HUD.
    /// Created automatically when Play starts, so any (even empty) scene works.
    /// </summary>
    public class SimView : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (Application.isBatchMode || FindAnyObjectByType<SimView>() != null) return;
            new GameObject("SignalSim").AddComponent<SimView>();
        }

        public int seed = 1;
        SimCore sim;
        int ctlIndex = 3;                 // PRAVAH
        float speed = 1f, acc, timeOfDay;
        bool paused, showPanel = true, showHelp;
        readonly float[] rates = { 700, 450, 650, 450 };
        float ambRate = 12f;          // demo rate (experiments use the scenario's own rate)
        bool firstAmbSent;            // one ambulance shortly after every (re)start, so the view always shows one
        readonly SimOptions opt = new SimOptions { cattlePerHour = 4f };

        WorldBuilder world;
        readonly CameraRig rig = new CameraRig();
        readonly Dictionary<Vehicle, VehicleRig> rigs = new Dictionary<Vehicle, VehicleRig>();
        readonly HashSet<Vehicle> alive = new HashSet<Vehicle>();
        readonly List<Vehicle> gone = new List<Vehicle>();
        Transform traffic;
        readonly float[] queueHist = new float[300];
        int histN; float histT;

        RunResult[] cmp;
        string cmpScenario;
        GUIStyle box, head, small, label, bar;
        Texture2D white;
        const float PanelW = 360f;

        void Start()
        {
            Application.targetFrameRate = 60;
            QualitySettings.shadowDistance = 160f;
            QualitySettings.pixelLightCount = 8;
            Setup();
        }

        void Setup()
        {
            world = new WorldBuilder();
            world.Build(transform);
            Post.Make(transform);
            traffic = new GameObject("Traffic").transform;
            traffic.SetParent(transform, false);
            rig.Init(transform, world);
            Restart();
        }

        void Restart()
        {
            foreach (var kv in rigs) Shapes.Kill(kv.Value.go);
            rigs.Clear();
            (sim?.controller as System.IDisposable)?.Dispose();
            sim = new SimCore(seed, Controllers.Make(ctlIndex), new SimOptions
            {
                freeLeft = opt.freeLeft, redJumping = opt.redJumping, busStops = opt.busStops, cattlePerHour = opt.cattlePerHour,
            });
            ApplyRates();
            histN = 0;
            firstAmbSent = false;
        }

        void ApplyRates()
        {
            for (int a = 0; a < 4; a++) sim.rate[a] = rates[a];
            sim.ambulancePerHour = ambRate;
            sim.opt.freeLeft = opt.freeLeft; sim.opt.redJumping = opt.redJumping; sim.opt.busStops = opt.busStops; sim.opt.cattlePerHour = opt.cattlePerHour;
        }

        bool MouseOverUi()
        {
            Vector2 m = Input.mousePosition; m.y = Screen.height - m.y;
            return (showPanel && m.x < PanelW + 20f) || m.y < 44f;
        }

        void Update()
        {
            Keys();
            if (!paused)
            {
                acc += Time.deltaTime * speed;
                int steps = 0;
                while (acc >= SimCore.DT && steps < 400) { sim.Step(); acc -= SimCore.DT; steps++; }
                if (!firstAmbSent && sim.t >= 20f) { sim.SpawnAmbulance(UnityEngine.Random.Range(0, 4)); firstAmbSent = true; }
                if (steps == 400) acc = 0f;   // too slow for this speed: drop the backlog instead of spiralling
                histT += steps * SimCore.DT;
                while (histT >= 2f)
                {
                    histT -= 2f;
                    float q = 0; for (int a = 0; a < 4; a++) q += sim.app[a].queued;
                    queueHist[histN % queueHist.Length] = q; histN++;
                }
            }
            SyncVehicles(paused ? 0f : Time.deltaTime * speed);
            SyncSignals();
            world.UpdatePedestrians(Time.deltaTime, Time.time);
            rig.Update(sim, Time.deltaTime, MouseOverUi());
            world.FollowRain((rig.mode == CamMode.Cctv ? rig.cctv[rig.cctvIndex] : rig.main).transform.position);
        }

        void Keys()
        {
            for (int i = 0; i < 8; i++) if (Input.GetKeyDown(KeyCode.Alpha1 + i)) rig.SetMode((CamMode)i);
            if (Input.GetKeyDown(KeyCode.H)) showPanel = !showPanel;
            if (Input.GetKeyDown(KeyCode.M)) rig.showMinimap = !rig.showMinimap;
            if (Input.GetKeyDown(KeyCode.V)) rig.driverView = !rig.driverView;
            if (Input.GetKeyDown(KeyCode.F)) { rig.SetMode(CamMode.Follow); rig.PickNext(sim); }
            if (Input.GetKeyDown(KeyCode.N)) SetTime(timeOfDay < 0.25f ? 0.55f : timeOfDay < 0.75f ? 1f : 0f);
            if (Input.GetKeyDown(KeyCode.R)) world.SetRain(!world.raining);
            if (Input.GetKeyDown(KeyCode.Space)) paused = !paused;
            if (Input.GetKeyDown(KeyCode.B)) sim.SpawnAmbulance(UnityEngine.Random.Range(0, 4));
            if (Input.GetKeyDown(KeyCode.F1)) showHelp = !showHelp;
            if (Input.GetMouseButtonDown(0) && !MouseOverUi() && rig.mode != CamMode.ControlRoom)
            {
                var v = rig.Pick(sim, Input.mousePosition);
                if (v != null) { rig.follow = v; rig.SetMode(CamMode.Follow); }
            }
        }

        void SetTime(float t) { timeOfDay = t; world.SetTimeOfDay(t); Post.TimeOfDay(t); }

        void SyncVehicles(float simDt)
        {
            alive.Clear();
            foreach (var v in sim.vehicles)
            {
                alive.Add(v);
                if (!rigs.TryGetValue(v, out var r)) rigs[v] = r = VehicleModels.Build(v, traffic);
                r.Update(v, simDt, world.raining ? Mathf.Max(timeOfDay, 0.5f) : timeOfDay, Time.time);   // lights on in the rain
            }
            gone.Clear();
            foreach (var kv in rigs) if (!alive.Contains(kv.Key)) gone.Add(kv.Key);
            foreach (var v in gone) { Shapes.Kill(rigs[v].go); rigs.Remove(v); }
        }

        void SyncSignals()
        {
            Color[] on = { new Color(1f, 0.1f, 0.08f), new Color(1f, 0.65f, 0f), new Color(0.1f, 1f, 0.35f) };
            for (int a = 0; a < 4; a++)
            {
                var lt = sim.LightFor(a);
                int k = lt == Light.Red ? 0 : lt == Light.Amber ? 1 : 2;
                for (int i = 0; i < 3; i++) world.lamps[a, i].sharedMaterial = i == k ? Mats.Glow(on[i], 4f) : Mats.Solid(on[i] * 0.12f, 0.8f);
                float left = Countdown(a);
                world.timers[a].text = left >= 0 ? Mathf.CeilToInt(left).ToString("00") : "--";
                world.timers[a].color = on[k];
            }
        }

        /// <summary>seconds until this approach's light changes; -1 when an adaptive controller has not decided yet</summary>
        float Countdown(int a)
        {
            if (a == sim.green && sim.phase == SimCore.Phase.Green) return Mathf.Max(0f, sim.greenDur - sim.phaseT);
            if (a == sim.green && sim.phase == SimCore.Phase.Amber) return SimCore.AmberSec - sim.phaseT;
            float[] g = null;
            if (sim.controller is FixedTimeController f) g = new[] { f.greenSec, f.greenSec, f.greenSec, f.greenSec };
            else if (sim.controller is WebsterController w) g = new[] { w.GreenOf(0), w.GreenOf(1), w.GreenOf(2), w.GreenOf(3) };
            if (g == null) return -1f;
            const float lost = SimCore.AmberSec + SimCore.AllRedSec;
            float t = sim.phase == SimCore.Phase.Green ? sim.greenDur - sim.phaseT + lost
                    : sim.phase == SimCore.Phase.Amber ? SimCore.AmberSec - sim.phaseT + SimCore.AllRedSec : SimCore.AllRedSec - sim.phaseT;
            for (int o = (sim.green + 1) % 4; o != a; o = (o + 1) % 4) t += g[o] + lost;
            return t;
        }

        // ---------------- edit-mode capture (Editor/Snapshot.cs) ----------------
        public void PrepareSnapshot(int ctl, float minutes, float tod, bool raining = false)
        {
            ctlIndex = ctl;
            Setup();
            for (int i = 0; i < minutes * 60f / SimCore.DT; i++)
            {
                if (i == (int)((minutes * 60f - 12f) / SimCore.DT)) sim.SpawnAmbulance(0);   // so the ambulance view has one
                sim.Step();
            }
            SetTime(tod);
            if (raining) world.SetRain(true);
            SyncVehicles(0.1f);
            SyncSignals();
            world.UpdatePedestrians(0f, 1f);
        }

        /// <summary>render one camera mode into a texture (several viewports for the control room)</summary>
        public Texture2D Capture(CamMode mode, int w, int h, float yaw = 25f)
        {
            rig.SetMode(mode);
            if (mode == CamMode.Follow || mode == CamMode.Ambulance)
            {
                rig.follow = null;
                foreach (var v in sim.vehicles)
                    if (mode == CamMode.Ambulance ? v.type == VType.Ambulance : v.type == VType.Car && !v.Crossed && v.DistToStop < 40f && v.DistToStop > 8f) { rig.follow = v; break; }
            }
            for (int i = 0; i < 120; i++) rig.Update(sim, 1f / 30f, true);   // let the smoothing settle
            if (mode == CamMode.Orbit)
            {
                var r = Quaternion.Euler(50f, yaw, 0);
                rig.main.transform.SetPositionAndRotation(r * new Vector3(0, 0, -110f), r);
            }
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            Grab(rig.main, tex, 0, 0, w, h);   // warm-up frame: shadows, probes and post-processing history settle
            if (mode == CamMode.ControlRoom)
                for (int a = 0; a < 4; a++) Grab(rig.cctv[a], tex, (a % 2) * w / 2, a < 2 ? h / 2 : 0, w / 2, h / 2);
            else Grab(mode == CamMode.Cctv ? rig.cctv[rig.cctvIndex] : rig.main, tex, 0, 0, w, h);
            tex.Apply();
            return tex;
        }

        static void Grab(Camera c, Texture2D dst, int x, int y, int w, int h)
        {
            var rt = new RenderTexture(w, h, 24);
            var rect = c.rect;
            c.rect = new Rect(0, 0, 1, 1); c.targetTexture = rt; c.Render(); c.targetTexture = null; c.rect = rect;
            RenderTexture.active = rt;
            dst.ReadPixels(new Rect(0, 0, w, h), x, y);
            RenderTexture.active = null;
            rt.Release();
        }

        // ---------------- HUD ----------------
        void Styles()
        {
            if (box != null) return;
            white = Texture2D.whiteTexture;
            box = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, padding = new RectOffset(10, 10, 8, 8), fontSize = 13 };
            box.normal.background = MakeTex(new Color(0.05f, 0.06f, 0.08f, 0.82f));
            head = new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold, fontSize = 15 };
            head.normal.textColor = new Color(1f, 0.82f, 0.4f);
            small = new GUIStyle(GUI.skin.label) { fontSize = 12, wordWrap = true };
            small.normal.textColor = new Color(0.9f, 0.9f, 0.9f);
            label = new GUIStyle(GUI.skin.label) { richText = true, fontSize = 13 };
            bar = new GUIStyle(GUI.skin.box);
            bar.normal.background = MakeTex(new Color(0.05f, 0.06f, 0.08f, 0.88f));
        }

        static Texture2D MakeTex(Color c)
        {
            var t = new Texture2D(1, 1); t.SetPixel(0, 0, c); t.Apply(); return t;
        }

        void OnDestroy() => (sim?.controller as System.IDisposable)?.Dispose();

        void OnGUI()
        {
            Styles();
            TopBar();
            if (rig.mode == CamMode.Cctv) CctvOverlay(rig.cctv[rig.cctvIndex], rig.cctvIndex);
            if (rig.mode == CamMode.ControlRoom) for (int a = 0; a < 4; a++) CctvOverlay(rig.cctv[a], a);
            if (rig.mode != CamMode.ControlRoom && rig.mode != CamMode.Cctv) ApproachLabels();
            if ((rig.mode == CamMode.Follow || rig.mode == CamMode.Ambulance) && rig.follow != null) FollowCard(rig.follow);
            if (rig.showMinimap && rig.mode != CamMode.ControlRoom)
                GUI.Label(new Rect(Screen.width * 0.79f, Screen.height * 0.02f - 2, 200, 20), "<b><color=white>MAP</color></b>", label);
            if (showPanel) Panel();
            if (showHelp) Help();
        }

        void TopBar()
        {
            GUILayout.BeginArea(new Rect(showPanel ? PanelW + 20 : 10, 6, Screen.width - (showPanel ? PanelW + 30 : 20), 36), bar);
            GUILayout.BeginHorizontal();
            int m = GUILayout.Toolbar((int)rig.mode, new[] { "1 Orbit", "2 Fly", "3 Top", "4 CCTV", "5 Control room", "6 Follow", "7 Ambulance", "8 Drone" }, GUILayout.Height(24));
            if (m != (int)rig.mode) rig.SetMode((CamMode)m);
            if (rig.mode == CamMode.Cctv && GUILayout.Button("Next cam", GUILayout.Height(24), GUILayout.Width(80))) rig.SetMode(CamMode.Cctv);
            GUILayout.Space(10);
            int tod = timeOfDay < 0.25f ? 0 : timeOfDay < 0.75f ? 1 : 2;
            int nt = GUILayout.Toolbar(tod, new[] { "Day", "Evening", "Night" }, GUILayout.Height(24), GUILayout.Width(190));
            if (nt != tod) SetTime(nt == 0 ? 0f : nt == 1 ? 0.55f : 1f);
            bool rn = GUILayout.Toggle(world.raining, " Monsoon (R)", GUILayout.Height(24), GUILayout.Width(110));
            if (rn != world.raining) world.SetRain(rn);
            if (GUILayout.Button(showPanel ? "Hide panel (H)" : "Panel (H)", GUILayout.Height(24), GUILayout.Width(110))) showPanel = !showPanel;
            if (GUILayout.Button("? (F1)", GUILayout.Height(24), GUILayout.Width(55))) showHelp = !showHelp;
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        void Help()
        {
            var r = new Rect(Screen.width / 2 - 230, Screen.height / 2 - 170, 460, 340);
            GUILayout.BeginArea(r, box);
            GUILayout.Label("Controls", head);
            GUILayout.Label(
                "1 Orbit - right-drag rotate, WASD / middle-drag pan, scroll zoom\n" +
                "2 Free fly - right-drag look, WASD move, Q/E down/up, Shift fast\n" +
                "3 Top-down - scroll zoom\n" +
                "4 CCTV - the junction cameras with what they detect; press 4 again for the next camera\n" +
                "5 Control room - all four CCTV feeds\n" +
                "6 Follow - click any vehicle (or F for the next); V toggles driver's seat / chase; scroll distance\n" +
                "7 Ambulance - chase the nearest ambulance\n" +
                "8 Drone - slow cinematic orbit\n\n" +
                "N day / evening / night    R monsoon rain    B send ambulance    Space pause    H panel    M map    F1 this help", small);
            if (GUILayout.Button("Close")) showHelp = false;
            GUILayout.EndArea();
        }

        void Panel()
        {
            GUILayout.BeginArea(new Rect(10, 6, PanelW, Screen.height - 12), box);
            scroll = GUILayout.BeginScrollView(scroll);
            GUILayout.Label("SMART SIGNAL SIMULATOR - INDIA", head);
            GUILayout.Label("4-arm junction, left-hand traffic, mixed traffic, one approach green at a time", small);

            GUILayout.Space(4);
#if UNITY_WEBGL
            var ctlNames = Controllers.Names[..Controllers.Bridge];   // a browser cannot open the bridge's TCP socket
#else
            var ctlNames = Controllers.Names;
#endif
            int c = GUILayout.SelectionGrid(ctlIndex, ctlNames, 2);
            if (c != ctlIndex) { ctlIndex = c; (sim.controller as System.IDisposable)?.Dispose(); sim.controller = Controllers.Make(ctlIndex); }
            GUILayout.BeginHorizontal();
            GUILayout.Label("Speed", small, GUILayout.Width(42));
            float[] sp = { 1, 2, 5, 10, 20 };
            int si = System.Array.IndexOf(sp, speed);
            int ns = GUILayout.Toolbar(si, new[] { "1x", "2x", "5x", "10x", "20x" });
            if (ns != si) speed = sp[ns];
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button(paused ? "Resume" : "Pause")) paused = !paused;
            if (GUILayout.Button("Restart (same traffic)")) Restart();
            GUILayout.EndHorizontal();

            GUILayout.Space(6);
            GUILayout.Label("Signal", head);
            string ph = sim.phase == SimCore.Phase.Green ? $"<color=#4f4>GREEN</color> {Geo.Names[sim.green]}  {sim.phaseT:F0}/{sim.greenDur:F0} s"
                      : sim.phase == SimCore.Phase.Amber ? $"<color=#fb0>AMBER</color> {Geo.Names[sim.green]}" : "<color=#f55>ALL RED</color>";
            GUILayout.Label($"<color=white>{sim.controller.Name}: {ph}</color>", label);
            GUILayout.Label(sim.reason, small);
            if (sim.boxBlockers != 0) GUILayout.Label("<color=#f55><b>JUNCTION BOX BLOCKED</b></color>", label);
            GUILayout.Label("Approach   queue   PCU   people   longest wait   served", small);
            for (int a = 0; a < 4; a++)
            {
                var s = sim.app[a];
                GUILayout.Label($"{Geo.Names[a],-8} {s.queued,5} {s.pcu,6:F1} {s.persons,6:F0} {s.maxWait,8:F0} s {s.crossed,7}{(s.ambulance ? "  AMB" : "")}", small);
            }

            GUILayout.Space(6);
            GUILayout.Label("Results so far", head);
            var st = sim.stats;
            float mins = Mathf.Max(0.01f, (sim.t - st.since) / 60f);
            sim.Backlog(out int bl, out float blw);
            GUILayout.Label($"Sim time {sim.t / 60f:F1} min   served {st.served}   ({st.served / mins * 60f:F0} veh/h)", small);
            GUILayout.Label($"Average wait {st.AvgWait:F1} s per vehicle, {st.PersonWait:F1} s per person   worst {st.maxWait:F0} s", small);
            GUILayout.Label($"Red-light jumps {st.violations} ({st.violations / mins * 60f:F0}/h)   box blocked {st.boxBlocked / Mathf.Max(1f, sim.t - st.since) * 100f:F0}% of the time", small);
            GUILayout.Label($"Ambulance average wait {(st.ambServed > 0 ? st.AmbAvgWait.ToString("F1") + " s" : "-")}  ({st.ambServed})   still queued {bl}", small);
            QueueChart();

            GUILayout.Space(6);
            GUILayout.Label("Traffic (vehicles / hour)", head);
            bool changed = false;
            for (int a = 0; a < 4; a++)
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label($"{Geo.Names[a]} {rates[a]:F0}", small, GUILayout.Width(95));
                float r = Mathf.Round(GUILayout.HorizontalSlider(rates[a], 0, 1500) / 50f) * 50f;
                if (r != rates[a]) { rates[a] = r; changed = true; }
                GUILayout.EndHorizontal();
            }
            GUILayout.BeginHorizontal();
            GUILayout.Label($"Ambulances/h {ambRate:F0}", small, GUILayout.Width(115));
            float ar = Mathf.Round(GUILayout.HorizontalSlider(ambRate, 0, 30));
            if (ar != ambRate) { ambRate = ar; changed = true; }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            GUILayout.Label($"Stray cattle/h {opt.cattlePerHour:F0}", small, GUILayout.Width(115));
            float cr = Mathf.Round(GUILayout.HorizontalSlider(opt.cattlePerHour, 0, 20));
            if (cr != opt.cattlePerHour) { opt.cattlePerHour = cr; changed = true; }
            GUILayout.EndHorizontal();
            bool fl = GUILayout.Toggle(opt.freeLeft, " Free left turn");
            bool rj = GUILayout.Toggle(opt.redJumping, " Red-light jumping (impatient riders)");
            bool bsx = GUILayout.Toggle(opt.busStops, " Bus stops / auto pickups after the junction");
            if (fl != opt.freeLeft || rj != opt.redJumping || bsx != opt.busStops) { opt.freeLeft = fl; opt.redJumping = rj; opt.busStops = bsx; changed = true; }
            int col = 0;
            GUILayout.BeginHorizontal();
            foreach (var s in Scenario.All)
            {
                if (GUILayout.Button(s.name, GUILayout.Width(165)))
                {
                    for (int a = 0; a < 4; a++) rates[a] = s.rates[a];
                    ambRate = s.ambPerHour; opt.busStops = s.busStops; opt.cattlePerHour = s.cattlePerHour; changed = true;
                }
                if (++col % 2 == 0) { GUILayout.EndHorizontal(); GUILayout.BeginHorizontal(); }
            }
            GUILayout.EndHorizontal();
            if (changed) ApplyRates();
            GUILayout.BeginHorizontal();
            GUILayout.Label("Send ambulance", small, GUILayout.Width(105));
            for (int a = 0; a < 4; a++) if (GUILayout.Button(Geo.Names[a].Substring(0, 1))) sim.SpawnAmbulance(a);
            GUILayout.EndHorizontal();

            GUILayout.Space(6);
            GUILayout.Label("Compare all controllers (same traffic)", head);
            if (GUILayout.Button("Run 15 sim-minutes with current settings"))
            {
                var sc = new Scenario("Current settings", rates[0], rates[1], rates[2], rates[3], ambRate, opt.busStops, opt.cattlePerHour);
                cmp = new RunResult[Controllers.Bridge];   // built-in controllers; the bridge is compared with Experiment -only
                for (int k = 0; k < cmp.Length; k++) cmp[k] = Compare.Run(Controllers.Make(k), sc, seed, 15f);
                cmpScenario = $"N {rates[0]:F0}  E {rates[1]:F0}  S {rates[2]:F0}  W {rates[3]:F0} veh/h, {ambRate:F0} amb/h, {opt.cattlePerHour:F0} cattle/h";
            }
            if (cmp != null)
            {
                GUILayout.Label(cmpScenario, small);
                GUILayout.Label("controller     wait  person  worst  amb  jumps/h", small);
                foreach (var r in cmp)
                    GUILayout.Label($"{r.controller,-12} {r.avgWait,5:F1} {r.personWait,6:F1} {r.maxWait,5:F0} {r.ambWait,5:F1} {r.violationsPerHour,5:F0}", small);
            }
            GUILayout.EndScrollView();
            GUILayout.EndArea();
        }

        Vector2 scroll;

        void QueueChart()
        {
            var r = GUILayoutUtility.GetRect(PanelW - 40, 60);
            GUI.color = new Color(1, 1, 1, 0.08f); GUI.DrawTexture(r, white); GUI.color = Color.white;
            int n = Mathf.Min(histN, queueHist.Length);
            if (n < 2) return;
            float max = 10f;
            for (int i = 0; i < n; i++) max = Mathf.Max(max, queueHist[(histN - n + i) % queueHist.Length]);
            float bw = r.width / queueHist.Length;
            GUI.color = new Color(1f, 0.75f, 0.3f, 0.9f);
            for (int i = 0; i < n; i++)
            {
                float v = queueHist[(histN - n + i) % queueHist.Length] / max * r.height;
                GUI.DrawTexture(new Rect(r.x + (queueHist.Length - n + i) * bw, r.yMax - v, Mathf.Max(1f, bw), v), white);
            }
            GUI.color = Color.white;
            GUI.Label(new Rect(r.x + 4, r.y, r.width, 18), $"vehicles queued, last 10 min (max {max:F0})", small);
        }

        void ApproachLabels()
        {
            for (int a = 0; a < 4; a++)
            {
                Vector3 w = Geo.U[a] * (Geo.J + 22f) + Vector3.up * 7f;
                Vector3 sp = rig.main.WorldToScreenPoint(w);
                if (sp.z <= 0) continue;
                var s = sim.app[a];
                var colr = sim.LightFor(a) == Light.Green ? "#3f3" : sim.LightFor(a) == Light.Amber ? "#fb0" : "#f55";
                GUI.Label(new Rect(sp.x - 60, Screen.height - sp.y - 12, 190, 40),
                          $"<b><color={colr}>{Geo.Names[a]}</color></b>\n<color=white>{s.queued} queued, ~{s.persons:F0} people</color>", label);
            }
        }

        /// <summary>what the junction camera reports to the controller: boxes on every vehicle it counts</summary>
        void CctvOverlay(Camera cam, int a)
        {
            Rect vp = cam.pixelRect;
            Rect gr = new Rect(vp.x, Screen.height - vp.yMax, vp.width, vp.height);
            var ist = System.DateTime.UtcNow.AddHours(5.5);
            GUI.Label(new Rect(gr.x + 8, gr.y + 46, 600, 22),
                      $"<b><color=white>CAM-0{a + 1}  {Geo.Names[a].ToUpper()} APPROACH   {ist:dd-MM-yyyy HH:mm:ss} IST</color></b>", label);
            var s = sim.app[a];
            GUI.Label(new Rect(gr.x + 8, gr.y + 66, 600, 22),
                      $"<color=#ffd24d>queue {s.queued}  |  {s.pcu:F1} PCU  |  ~{s.persons:F0} people  |  longest wait {s.maxWait:F0} s{(s.ambulance ? "  |  AMBULANCE" : "")}</color>", label);
            foreach (var v in sim.vehicles)
            {
                if (v.approach != a || v.Obstacle || v.Crossed || v.DistToStop > sim.opt.detectRange) continue;
                if (!Bounds(cam, v, out Rect br) || !gr.Overlaps(br)) continue;
                Color c = v.type == VType.Ambulance ? Color.red : v.jumper ? new Color(1f, 0.3f, 0.9f) : sim.Controlled(v) ? new Color(0.2f, 1f, 0.4f) : new Color(0.5f, 0.8f, 1f);
                Outline(br, c);
                string tag = v.type == VType.Ambulance ? "AMBULANCE" : v.jumper ? $"{v.type} RED-JUMP" : $"{v.type} {v.spec.pcu:0.#}";
                GUI.color = c;
                GUI.Label(new Rect(br.x, br.y - 16, 160, 18), tag, small);
                GUI.color = Color.white;
            }
        }

        static bool Bounds(Camera cam, Vehicle v, out Rect r)
        {
            r = default;
            Vector3 f = v.dir, l = Geo.Left(v.dir);
            float xmin = float.MaxValue, ymin = float.MaxValue, xmax = float.MinValue, ymax = float.MinValue;
            for (int i = 0; i < 8; i++)
            {
                Vector3 p = v.pos + f * ((i & 1) == 0 ? -0.5f : 0.5f) * v.spec.len + l * ((i & 2) == 0 ? -0.5f : 0.5f) * v.spec.wid + Vector3.up * ((i & 4) == 0 ? 0f : v.spec.height + 0.3f);
                var sp = cam.WorldToScreenPoint(p);
                if (sp.z <= 0) return false;
                xmin = Mathf.Min(xmin, sp.x); xmax = Mathf.Max(xmax, sp.x); ymin = Mathf.Min(ymin, sp.y); ymax = Mathf.Max(ymax, sp.y);
            }
            r = new Rect(xmin, Screen.height - ymax, xmax - xmin, ymax - ymin);
            return r.width > 4 && r.height > 4;
        }

        void Outline(Rect r, Color c)
        {
            GUI.color = c;
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 2), white);
            GUI.DrawTexture(new Rect(r.x, r.yMax - 2, r.width, 2), white);
            GUI.DrawTexture(new Rect(r.x, r.y, 2, r.height), white);
            GUI.DrawTexture(new Rect(r.xMax - 2, r.y, 2, r.height), white);
            GUI.color = Color.white;
        }

        void FollowCard(Vehicle v)
        {
            var r = new Rect(Screen.width - 300, Screen.height - 150, 290, 140);
            GUILayout.BeginArea(r, box);
            string plate = rigs.TryGetValue(v, out var vr) && vr.plate != null ? vr.plate : "-";
            GUILayout.Label($"{v.type}  {plate}", head);
            GUILayout.Label($"{Geo.Names[v.approach]} approach, turning {v.turn.ToString().ToLower()}   {v.v * 3.6f:F0} km/h", small);
            GUILayout.Label(v.Crossed ? $"Through the signal after waiting {v.wait:F0} s" : $"{Mathf.Max(0, v.DistToStop):F0} m to the stop line, waited {v.wait:F0} s", small);
            if (v.jumper) GUILayout.Label("<color=#f6c>Jumped the red light</color>", label);
            GUILayout.Label(rig.driverView ? "Driver's seat (V for chase cam)" : "Chase cam (V for driver's seat)", small);
            GUILayout.EndArea();
        }
    }
}
