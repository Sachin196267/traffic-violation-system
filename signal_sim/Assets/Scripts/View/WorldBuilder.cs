using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>Builds the Indian street around the junction and runs time of day and pedestrians.</summary>
    public class WorldBuilder
    {
        public Transform root;
        public readonly Renderer[,] lamps = new Renderer[4, 3];   // signal heads [approach, red/amber/green]
        public readonly TextMesh[] timers = new TextMesh[4];
        public readonly Vector3[] cctvPos = new Vector3[4], cctvLook = new Vector3[4];
        UnityEngine.Light sun;
        Material sky;
        ReflectionProbe probe;
        readonly Dictionary<string, Material> skies = new Dictionary<string, Material>();
        SkyInfo skyInfo;

        [System.Serializable] class Sun { public string id; public float azimuth, elevation, peak; }
        [System.Serializable] class SkyInfo { public Sun day, evening, night, rain; }
        public bool raining;
        ParticleSystem rain;
        readonly List<(Material m, Color tint, float smooth, bool alphaSmooth)> wettable = new List<(Material, Color, float, bool)>();
        readonly List<UnityEngine.Light> streetLights = new List<UnityEngine.Light>();
        readonly List<Renderer> streetHeads = new List<Renderer>();
        readonly List<Material> facades = new List<Material>();
        readonly List<Walker> walkers = new List<Walker>();
        public float night;

        const float W = Geo.W, L = Geo.L, J = Geo.J, Half = 2 * Geo.W, Foot = 3f;
        static readonly string[] Shops =
        {
            "SRI GANESH STORES", "ANNAPURNA TIFFIN", "GUPTA MEDICALS", "ROYAL TAILORS", "BALAJI HARDWARE", "NEW KRISHNA SWEETS",
            "LAXMI MOBILE POINT", "CHAI KATTA", "SHREE XEROX", "OM SAI KIRANA", "AMAN ELECTRICALS", "MAA DURGA FOOTWEAR",
            "SAGAR PURE VEG", "PATEL PROVISION", "NOOR BAKERY", "CITY CLINIC", "JAI HIND OPTICALS", "STAR CYCLE MART",
        };

        static Material M(Color c, float gloss = 0.1f) => Mats.Solid(c, gloss);

        public void Build(Transform parent)
        {
            root = new GameObject("World").transform;
            root.SetParent(parent, false);
            var rnd = new System.Random(5);
            Lighting();

            var asphaltNS = Mats.Pbr("asphalt", new Vector2(2 * Half / 3f, 2 * L / 3f), new Color(0.78f, 0.78f, 0.78f), 0.9f, 0.8f);
            var asphaltEW = Mats.Pbr("asphalt", new Vector2(2 * L / 3f, 2 * Half / 3f), new Color(0.78f, 0.78f, 0.78f), 0.9f, 0.8f);
            Shapes.Box(root, new Vector3(0, -0.06f, 0), new Vector3(2400f, 0.1f, 2400f), Mats.Pbr("ground", new Vector2(600f, 600f), new Color(0.95f, 0.86f, 0.74f), 0.6f));
            foreach (var wm in new[] { asphaltNS, asphaltEW, Mats.Pbr("ground", new Vector2(600f, 600f), new Color(0.95f, 0.86f, 0.74f), 0.6f) })
                Wettable(wm);
            Shapes.Box(root, Vector3.zero, new Vector3(2 * Half, 0.02f, 2 * L), asphaltNS);
            Shapes.Box(root, new Vector3(0, 0.001f, 0), new Vector3(2 * L, 0.02f, 2 * Half), asphaltEW);
            var paint = M(new Color(0.92f, 0.92f, 0.88f), 0.3f);

            for (int a = 0; a < 4; a++)
            {
                Vector3 u = Geo.U[a], d = -u, lf = Geo.Left(d);
                var rot = Quaternion.LookRotation(d).eulerAngles;
                float armLen = L - Half - Foot;
                float armMid = (L + Half + Foot) * 0.5f;
                var paving = Mats.Pbr("pavers", new Vector2(Foot / 1.5f, armLen / 1.5f), new Color(0.92f, 0.86f, 0.8f), 0.8f);
                Wettable(paving);
                var kerb = Mats.Textured("kerb", Mats.Kerb, new Vector2(armLen, 1f), Color.white, 0.2f);

                foreach (float side in new[] { -1f, 1f })
                {
                    Shapes.Box(root, u * armMid + lf * side * (Half + Foot / 2) + Vector3.up * 0.08f, new Vector3(Foot, 0.16f, armLen), paving, rot);
                    Shapes.Box(root, u * armMid + lf * side * (Half + 0.15f) + Vector3.up * 0.12f, new Vector3(0.3f, 0.25f, armLen), kerb, rot);
                }
                // central median with shrubs and street lights, stopping short of the junction
                float m0 = J + 6f, mLen = L - m0;
                Shapes.Box(root, u * (m0 + mLen / 2) + Vector3.up * 0.13f, new Vector3(0.8f, 0.26f, mLen), Mats.Textured("kerbMedian", Mats.Kerb, new Vector2(mLen, 1f), Color.white, 0.2f), rot);
                for (float s = m0 + 3f; s < L - 2f; s += 5f)
                    Shapes.Part(root, PrimitiveType.Sphere, u * s + Vector3.up * 0.55f, new Vector3(0.7f, 0.6f, 0.9f), M(new Color(0.18f, 0.42f, 0.16f)), default, false);
                for (float s = m0 + 8f; s < L; s += 32f) StreetLight(u * s, u, s < 60f);

                Shapes.Box(root, u * (L + J) * 0.5f + Vector3.up * 0.02f, new Vector3(0.15f, 0.01f, L - J), paint, rot, false);
                foreach (float side in new[] { -1f, 1f })
                    for (float s = J + 3f; s < L; s += 6f)
                        Shapes.Box(root, u * s + lf * side * W + Vector3.up * 0.02f, new Vector3(0.12f, 0.01f, 3f), paint, rot, false);
                Shapes.Box(root, u * J + lf * W + Vector3.up * 0.02f, new Vector3(2 * W, 0.01f, 0.4f), paint, rot, false);
                for (float x = -Half + 0.5f; x < Half; x += 1.2f)
                    Shapes.Box(root, u * (J - 1.4f) + lf * x + Vector3.up * 0.02f, new Vector3(0.6f, 0.01f, 2.2f), paint, rot, false);
                // turn arrows painted in each lane
                Shapes.Box(root, u * (J + 12f) + lf * (1.5f * W) + Vector3.up * 0.021f, new Vector3(0.3f, 0.01f, 3f), paint, rot, false);
                Shapes.Box(root, u * (J + 10.8f) + lf * (1.5f * W + 0.5f) + Vector3.up * 0.021f, new Vector3(1.0f, 0.01f, 0.3f), paint, rot, false);
                Shapes.Box(root, u * (J + 12f) + lf * (0.5f * W) + Vector3.up * 0.021f, new Vector3(0.3f, 0.01f, 3f), paint, rot, false);

                SignalHead(a, u, d, lf);
                BusShelter(u, rnd);
                Frontage(a, u, lf, rnd);
                for (float s = Half + 10f; s < L; s += 14f)
                    foreach (float side in new[] { -1f, 1f })
                        if (rnd.NextDouble() < 0.6) Tree(u * s + lf * side * (Half + Foot - 0.6f), rnd);
                // parked two-wheelers on the footpath outside shops
                for (int i = 0; i < 6; i++)
                {
                    float s = 30f + (float)rnd.NextDouble() * 90f, side = rnd.NextDouble() < 0.5 ? -1f : 1f;
                    for (int k = 0; k < 3; k++)
                        VehicleModels.ParkedBike(root, u * (s + k * 0.9f) + lf * side * (Half + Foot - 0.9f) + Vector3.up * 0.16f,
                                                 Quaternion.LookRotation(lf * side).eulerAngles.y + 15f, rnd.Next(100));
                }
                // CCTV on the far-side corner, looking down the approach at oncoming fronts (ANPR view)
                cctvPos[a] = u * (J - 2f) + Geo.Left(u) * (Half + 1.5f) + Vector3.up * 7.5f;
                cctvLook[a] = u * 45f + lf * W;
                Shapes.Box(root, new Vector3(cctvPos[a].x, 3.75f, cctvPos[a].z), new Vector3(0.2f, 7.5f, 0.2f), M(new Color(0.3f, 0.3f, 0.3f)));
                Shapes.Box(root, cctvPos[a], new Vector3(0.3f, 0.3f, 0.6f), M(new Color(0.9f, 0.9f, 0.9f), 0.5f), Quaternion.LookRotation(cctvLook[a] - cctvPos[a]).eulerAngles);
            }
            Corners(rnd);
            Skyline(rnd);
            Billboards();
            Pedestrians(rnd);
            SetTimeOfDay(0f);
        }

        void Lighting()
        {
            sun = new GameObject("Sun").AddComponent<UnityEngine.Light>();
            sun.transform.SetParent(root, false);
            sun.type = LightType.Directional; sun.shadows = LightShadows.Soft; sun.shadowStrength = 0.8f;
            // photographed HDRI skies (Resources/PH, CC0) light the scene; procedural sky if they were not fetched
            var meta = Resources.Load<TextAsset>("PH/sky");
            var pano = Shader.Find("Skybox/Panoramic");
            if (meta != null && pano != null)
            {
                skyInfo = JsonUtility.FromJson<SkyInfo>(meta.text);
                foreach (var tod in new[] { "day", "evening", "night", "rain" })
                {
                    var tex = Resources.Load<Texture>($"PH/sky_{tod}");
                    if (tex == null) continue;
                    var m = new Material(pano); m.SetTexture("_MainTex", tex); m.SetFloat("_Mapping", 1f); m.SetFloat("_ImageType", 0f);
                    skies[tod] = m;
                }
            }
            var sh = Shader.Find("Skybox/Procedural");
            if (sh != null) { sky = new Material(sh); sky.SetFloat("_AtmosphereThickness", 1.25f); sky.SetFloat("_SunSize", 0.035f); RenderSettings.skybox = sky; }
            RenderSettings.defaultReflectionMode = UnityEngine.Rendering.DefaultReflectionMode.Skybox;
            probe = new GameObject("Street reflections").AddComponent<ReflectionProbe>();
            probe.transform.SetParent(root, false);
            probe.transform.position = new Vector3(0, 4f, 0);
            probe.mode = UnityEngine.Rendering.ReflectionProbeMode.Realtime;
            probe.refreshMode = UnityEngine.Rendering.ReflectionProbeRefreshMode.ViaScripting;
            probe.timeSlicingMode = UnityEngine.Rendering.ReflectionProbeTimeSlicingMode.NoTimeSlicing;
            probe.size = new Vector3(320f, 80f, 320f); probe.resolution = 256; probe.hdr = true; probe.farClipPlane = 400f;
            RenderSettings.sun = sun;
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
        }

        /// <summary>0 = noon, 0.5 = evening, 1 = night</summary>
        public void SetTimeOfDay(float n)
        {
            night = n;
            string tod = n < 0.25f ? "day" : n < 0.75f ? "evening" : "night";
            if (raining && tod != "night" && skies.ContainsKey("rain")) tod = "rain";
            if (skies.TryGetValue(tod, out var hdri)) { SetHdriTime(tod, hdri, n); return; }
            sun.transform.rotation = Quaternion.Euler(Mathf.Lerp(52f, 8f, Mathf.Min(1f, n * 2f)) + (n > 0.5f ? (n - 0.5f) * 80f : 0f), 35f + n * 160f, 0f);
            sun.intensity = Mathf.Lerp(1.2f, 0.08f, n);
            sun.color = Color.Lerp(new Color(1f, 0.96f, 0.88f), n < 0.75f ? new Color(1f, 0.6f, 0.35f) : new Color(0.55f, 0.65f, 1f), Mathf.Min(1f, n * 1.6f));
            RenderSettings.ambientSkyColor = Color.Lerp(new Color(0.62f, 0.66f, 0.72f), new Color(0.13f, 0.14f, 0.22f), n);
            RenderSettings.ambientEquatorColor = Color.Lerp(new Color(0.58f, 0.55f, 0.5f), new Color(0.11f, 0.1f, 0.12f), n);
            RenderSettings.ambientGroundColor = Color.Lerp(new Color(0.4f, 0.36f, 0.3f), new Color(0.03f, 0.03f, 0.04f), n);
            RenderSettings.fogColor = Color.Lerp(new Color(0.78f, 0.78f, 0.76f), new Color(0.06f, 0.07f, 0.1f), n);
            RenderSettings.fogDensity = Mathf.Lerp(0.0035f, 0.006f, n);
            if (sky) sky.SetFloat("_Exposure", Mathf.Lerp(1.25f, 0.05f, n));
            bool lightsOn = n > 0.3f;
            foreach (var l in streetLights) { l.enabled = lightsOn; l.intensity = Mathf.Lerp(0.8f, 2.2f, n); }
            var headMat = lightsOn ? Mats.Glow(new Color(1f, 0.85f, 0.55f), 3f) : M(new Color(0.85f, 0.85f, 0.8f), 0.6f);
            foreach (var r in streetHeads) r.sharedMaterial = headMat;
            foreach (var f in facades) f.SetColor("_EmissionColor", new Color(1f, 0.85f, 0.6f) * Mathf.Clamp01((n - 0.25f) * 1.6f) * 1.4f);
            DynamicGI.UpdateEnvironment();
        }

        /// <summary>sky photo + matching sun: direction from where the HDRI's sun is (sky.json), so shadows agree with the sky</summary>
        void SetHdriTime(string tod, Material hdri, float n)
        {
            RenderSettings.skybox = hdri;
            if (tod == "night")
            {   // moonlit ambient set directly: the skybox-derived one is too slow to recompute and too dim to see the street
                RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
                RenderSettings.ambientSkyColor = new Color(0.1f, 0.12f, 0.2f);
                RenderSettings.ambientEquatorColor = new Color(0.07f, 0.07f, 0.1f);
                RenderSettings.ambientGroundColor = new Color(0.04f, 0.035f, 0.035f);
            }
            else { RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Skybox; RenderSettings.ambientIntensity = 1f; }
            var info = tod == "day" ? skyInfo.day : tod == "evening" ? skyInfo.evening : tod == "rain" ? skyInfo.rain : skyInfo.night;
            float el = tod == "night" ? 40f : Mathf.Max(info.elevation, 6f);     // night: moonlight from above, not the horizon glow
            float phi = (0.5f - info.azimuth / 360f) * 2f * Mathf.PI;          // Skybox/Panoramic: u = 0.5 - atan2(z, x) / 2pi
            Vector3 toSun = new Vector3(Mathf.Cos(phi) * Mathf.Cos(el * Mathf.Deg2Rad), Mathf.Sin(el * Mathf.Deg2Rad), Mathf.Sin(phi) * Mathf.Cos(el * Mathf.Deg2Rad));
            sun.transform.rotation = Quaternion.LookRotation(-toSun);
            sun.intensity = tod == "day" ? 2.2f : tod == "evening" ? 1.1f : tod == "rain" ? 0.45f : 0.12f;
            sun.shadowStrength = tod == "rain" ? 0.35f : 0.8f;
            sun.color = tod == "day" ? new Color(1f, 0.95f, 0.86f) : tod == "evening" ? new Color(1f, 0.62f, 0.36f) : tod == "rain" ? new Color(0.85f, 0.9f, 1f) : new Color(0.6f, 0.7f, 1f);
            hdri.SetFloat("_Exposure", tod == "night" ? 0.35f : tod == "evening" ? 1.1f : tod == "rain" ? 0.9f : 1f);
            RenderSettings.fogColor = tod == "day" ? new Color(0.76f, 0.78f, 0.8f) : tod == "evening" ? new Color(0.62f, 0.5f, 0.42f)
                                    : tod == "rain" ? new Color(0.52f, 0.55f, 0.58f) : new Color(0.04f, 0.05f, 0.08f);
            RenderSettings.fogDensity = raining ? (tod == "night" ? 0.008f : 0.0065f) : tod == "day" ? 0.0022f : tod == "evening" ? 0.003f : 0.004f;
            Wet(raining);
            CityLights(n);
            DynamicGI.UpdateEnvironment();
            if (probe) probe.RenderProbe();
        }

        void Wettable(Material m)
        {
            if (m == null || wettable.Exists(w => w.m == m)) return;
            wettable.Add((m, m.color, m.HasProperty("_Smoothness") ? m.GetFloat("_Smoothness") : m.GetFloat("_Glossiness"), m.IsKeywordEnabled("_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A")));
        }

        /// <summary>wet monsoon surfaces: darker, glossy, reflecting the street; restored when the rain stops</summary>
        void Wet(bool wet)
        {
            foreach (var (m, tint, smooth, alphaSmooth) in wettable)
            {
                m.color = wet ? tint * 0.62f : tint;
                if (wet) m.DisableKeyword("_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A"); else if (alphaSmooth) m.EnableKeyword("_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A");
                m.SetFloat("_SmoothnessTextureChannel", wet ? 0f : alphaSmooth ? 1f : 0f);
                float v = wet ? 0.8f : smooth;
                if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", v);
                if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", v);
            }
        }

        public void SetRain(bool on)
        {
            raining = on;
            if (on && rain == null) rain = MakeRain();
            if (rain != null) { if (on) rain.Play(); else { rain.Stop(); rain.Clear(); } rain.gameObject.SetActive(on); }
            SetTimeOfDay(night);
        }

        /// <summary>keep the rain volume around the viewer</summary>
        public void FollowRain(Vector3 camPos)
        {
            if (rain != null && raining) rain.transform.position = camPos + Vector3.up * 14f;
        }

        ParticleSystem MakeRain()
        {
            var go = new GameObject("Monsoon rain");
            go.transform.SetParent(root, false);
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            var main = ps.main;
            main.loop = true; main.startLifetime = 1.1f; main.startSpeed = 0f; main.startSize = 0.025f;
            main.maxParticles = 12000; main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.startColor = new Color(0.75f, 0.8f, 0.85f, 0.5f);
            var em = ps.emission; em.rateOverTime = 9000f;
            var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Box; sh.scale = new Vector3(70f, 1f, 70f);
            var vel = ps.velocityOverLifetime; vel.enabled = true; vel.space = ParticleSystemSimulationSpace.World;
            vel.x = new ParticleSystem.MinMaxCurve(-1.5f, -0.5f); vel.y = new ParticleSystem.MinMaxCurve(-24f, -20f); vel.z = new ParticleSystem.MinMaxCurve(0.5f, 1.5f);
            var r = go.GetComponent<ParticleSystemRenderer>();
            r.renderMode = ParticleSystemRenderMode.Stretch; r.velocityScale = 0.035f; r.lengthScale = 1f;
            var shader = Shader.Find("SignalSim/Rain");
            r.sharedMaterial = shader != null ? new Material(shader) : Mats.Solid(new Color(0.8f, 0.85f, 0.9f));
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            return ps;
        }

        void CityLights(float n)
        {
            bool lightsOn = n > 0.3f || raining;
            foreach (var l in streetLights) { l.enabled = lightsOn; l.intensity = Mathf.Lerp(2f, 6f, Mathf.Max(n, raining ? 0.4f : 0f)); }
            var headMat = lightsOn ? Mats.Glow(new Color(1f, 0.85f, 0.55f), 6f) : M(new Color(0.85f, 0.85f, 0.8f), 0.6f);
            foreach (var r in streetHeads) r.sharedMaterial = headMat;
            foreach (var f in facades) f.SetColor("_EmissionColor", new Color(1f, 0.82f, 0.55f) * Mathf.Clamp01((n - 0.25f) * 1.6f) * 2.2f);
        }

        void SignalHead(int a, Vector3 u, Vector3 d, Vector3 lf)
        {
            Vector3 pole = u * (J + 0.5f) + lf * (Half + 1.2f);
            var rot = Quaternion.LookRotation(d).eulerAngles;
            var dark = M(new Color(0.08f, 0.08f, 0.08f), 0.3f);
            Shapes.Part(root, PrimitiveType.Cylinder, pole + Vector3.up * 3f, new Vector3(0.22f, 3f, 0.22f), M(new Color(0.2f, 0.2f, 0.2f), 0.4f));
            Shapes.Box(root, pole + Vector3.up * 0.4f, new Vector3(0.3f, 0.8f, 0.3f), Mats.Textured("kerbPole", Mats.Kerb, new Vector2(1, 4), Color.white), new Vector3(0, 0, 90));
            Shapes.Box(root, pole + Vector3.up * 4.4f, new Vector3(0.7f, 2.1f, 0.6f), dark, rot);
            Shapes.Box(root, pole + Vector3.up * 6.1f, new Vector3(1.4f, 0.9f, 0.3f), dark, rot);
            for (int k = 0; k < 3; k++)
            {
                lamps[a, k] = Shapes.Part(root, PrimitiveType.Sphere, pole + Vector3.up * (5.05f - k * 0.65f) + u * 0.35f, Vector3.one * 0.5f, dark, default, false).GetComponent<Renderer>();
                Shapes.Box(root, pole + Vector3.up * (5.3f - k * 0.65f) + u * 0.45f, new Vector3(0.55f, 0.05f, 0.25f), dark, rot, false);   // visor
            }
            timers[a] = Shapes.Text(root, "--", pole + Vector3.up * 6.1f + u * 0.2f, 0.6f, Color.red, rot);
        }

        void StreetLight(Vector3 pos, Vector3 u, bool withLight)
        {
            var grey = M(new Color(0.45f, 0.45f, 0.47f), 0.4f);
            Shapes.Part(root, PrimitiveType.Cylinder, pos + Vector3.up * 4.5f, new Vector3(0.18f, 4.5f, 0.18f), grey);
            var lf = Geo.Left(-u);
            foreach (float s in new[] { -1f, 1f })
            {
                Shapes.Box(root, pos + lf * s * 1.1f + Vector3.up * 8.9f, new Vector3(0.08f, 0.08f, 2.2f), grey, Quaternion.LookRotation(lf).eulerAngles);
                var head = Shapes.Box(root, pos + lf * s * 2.1f + Vector3.up * 8.8f, new Vector3(0.5f, 0.12f, 0.9f), grey, Quaternion.LookRotation(lf).eulerAngles, false);
                streetHeads.Add(head.GetComponent<Renderer>());
                if (!withLight) continue;
                var l = new GameObject("Street light").AddComponent<UnityEngine.Light>();
                l.transform.SetParent(root, false);
                l.transform.position = pos + lf * s * 2.1f + Vector3.up * 8.5f;
                l.type = LightType.Point; l.range = 22f; l.color = new Color(1f, 0.78f, 0.5f);
                streetLights.Add(l);
            }
        }

        void BusShelter(Vector3 u, System.Random rnd)
        {
            Vector3 c = u * 29f + Geo.Left(u) * (Half + Foot / 2 + 0.3f);
            var rot = Quaternion.LookRotation(u).eulerAngles;
            Shapes.Box(root, c + Vector3.up * 2.7f, new Vector3(2.4f, 0.12f, 6f), M(new Color(0.15f, 0.4f, 0.7f), 0.5f), rot);
            Shapes.Box(root, c + Geo.Left(u) * 1.1f + Vector3.up * 1.4f, new Vector3(0.05f, 2.5f, 6f), Mats.Solid(new Color(0.6f, 0.7f, 0.75f), 0.9f), rot);
            Shapes.Box(root, c + Geo.Left(u) * 0.7f + Vector3.up * 0.55f, new Vector3(0.4f, 0.08f, 5f), M(new Color(0.5f, 0.5f, 0.5f), 0.5f), rot);
            foreach (float e in new[] { -2.8f, 2.8f })
                Shapes.Box(root, c + u * e + Vector3.up * 1.35f, new Vector3(0.1f, 2.7f, 0.1f), M(new Color(0.3f, 0.3f, 0.3f), 0.5f));
            Shapes.Text(root, "BUS STOP", c + Vector3.up * 3.0f - Geo.Left(u) * 1.25f, 0.35f, Color.white, Quaternion.LookRotation(Geo.Left(u)).eulerAngles);
        }

        /// <summary>a row of shop-houses facing the road on both sides of an arm</summary>
        void Frontage(int a, Vector3 u, Vector3 lf, System.Random rnd)
        {
            Color[] walls = { new Color(0.95f, 0.82f, 0.66f), new Color(0.93f, 0.74f, 0.75f), new Color(0.97f, 0.9f, 0.6f),
                              new Color(0.72f, 0.87f, 0.82f), new Color(0.92f, 0.92f, 0.9f), new Color(0.85f, 0.68f, 0.55f), new Color(0.75f, 0.8f, 0.92f) };
            Color[] awn = { new Color(0.8f, 0.15f, 0.1f), new Color(0.1f, 0.45f, 0.75f), new Color(0.95f, 0.6f, 0.1f), new Color(0.2f, 0.55f, 0.25f) };
            foreach (float side in new[] { -1f, 1f })
            {
                float s = 25f;
                while (s < L - 8f)
                {
                    float w = 7f + (float)rnd.NextDouble() * 6f, depth = 11f + (float)rnd.NextDouble() * 4f;
                    int floors = 2 + rnd.Next(5);
                    float h = floors * 3.2f, front = Half + Foot + 0.5f;
                    Vector3 axis = u * (s + w / 2);
                    Vector3 toRoad = -lf * side;
                    var rot = Quaternion.LookRotation(u).eulerAngles;
                    var fac = Mats.Facade(walls[rnd.Next(walls.Length)], w / 3.2f, floors, rnd.Next(100));
                    facades.Add(fac);
                    Shapes.Box(root, axis + lf * side * (front + depth / 2) + Vector3.up * h / 2, new Vector3(depth, h, w - 0.3f), fac, rot);
                    Shapes.Box(root, axis + lf * side * (front + depth / 2) + Vector3.up * (h + 0.15f), new Vector3(depth + 0.2f, 0.3f, w - 0.1f), Mats.Pbr("concrete", new Vector2(3f, 3f), new Color(0.85f, 0.83f, 0.8f), 0.7f), rot);
                    Shapes.Part(root, PrimitiveType.Cylinder, axis + lf * side * (front + depth * 0.7f) + Vector3.up * (h + 1.0f), new Vector3(1.4f, 0.7f, 1.4f), M(new Color(0.06f, 0.06f, 0.06f), 0.3f));   // water tank
                    FacadeDetail(axis, lf * side, front, w, floors, rot, rnd);
                    // shop: shutter, awning, signboard facing the road
                    Shapes.Box(root, axis + lf * side * (front - 0.02f) + Vector3.up * 1.5f, new Vector3(0.05f, 2.8f, w * 0.85f),
                               rnd.NextDouble() < 0.5 ? M(new Color(0.55f, 0.57f, 0.6f), 0.4f) : Mats.Solid(new Color(0.3f, 0.35f, 0.38f), 0.9f), rot);
                    Shapes.Box(root, axis + lf * side * (front - 0.6f) + Vector3.up * 3.05f, new Vector3(1.2f, 0.06f, w * 0.9f), M(awn[rnd.Next(awn.Length)], 0.2f),
                               rot + new Vector3(0, 0, side * (Vector3.Dot(lf, Vector3.Cross(u, Vector3.up)) > 0 ? 12f : -12f)));
                    Shapes.Box(root, axis + lf * side * (front - 0.06f) + Vector3.up * 3.75f, new Vector3(0.08f, 0.9f, w * 0.85f), M(awn[rnd.Next(awn.Length)] * 0.9f, 0.3f), rot);
                    Shapes.Text(root, Shops[rnd.Next(Shops.Length)], axis + lf * side * (front - 0.12f) + Vector3.up * 3.75f, 0.42f, Color.white,
                                Quaternion.LookRotation(-toRoad).eulerAngles);
                    s += w + (rnd.NextDouble() < 0.2 ? 4f : 0f);
                }
            }
        }

        /// <summary>what makes an Indian shop-house: balconies with railings on upper floors and window AC units</summary>
        void FacadeDetail(Vector3 axis, Vector3 outward, float front, float w, int floors, Vector3 rot, System.Random rnd)
        {
            var slab = Mats.Pbr("concrete", new Vector2(1f, 2f), new Color(0.82f, 0.8f, 0.77f), 0.6f);
            var rail = M(new[] { new Color(0.15f, 0.15f, 0.16f), new Color(0.2f, 0.35f, 0.3f), new Color(0.55f, 0.55f, 0.55f) }[rnd.Next(3)], 0.5f);
            var ac = M(new Color(0.88f, 0.88f, 0.86f), 0.4f);
            bool balconies = rnd.NextDouble() < 0.7;
            for (int f = 1; f < floors; f++)
            {
                float y = f * 3.2f;
                if (balconies && (f % 2 == 1 || rnd.NextDouble() < 0.4))
                {
                    float bw = w * (0.45f + (float)rnd.NextDouble() * 0.35f), off = ((float)rnd.NextDouble() - 0.5f) * (w - bw) * 0.8f;
                    Vector3 along = Quaternion.Euler(rot) * Vector3.forward;
                    Vector3 c = axis + along * off + outward * (front - 0.6f);
                    Shapes.Box(root, c + Vector3.up * (y + 0.06f), new Vector3(1.2f, 0.12f, bw), slab, rot);
                    Shapes.Box(root, c - outward * 0.58f + Vector3.up * (y + 0.6f), new Vector3(0.05f, 1.0f, bw), rail, rot, false);
                }
                for (int k = 0; k < 2; k++)
                    if (rnd.NextDouble() < 0.35)
                    {
                        Vector3 along = Quaternion.Euler(rot) * Vector3.forward;
                        Vector3 c = axis + along * (((float)rnd.NextDouble() - 0.5f) * (w - 1.5f)) + outward * (front - 0.18f);
                        Shapes.Box(root, c + Vector3.up * (y + 1.0f + (float)rnd.NextDouble() * 0.8f), new Vector3(0.35f, 0.5f, 0.8f), ac, rot, false);
                    }
            }
        }

        void Corners(System.Random rnd)
        {
            // small parks with trees at the four corners (the plots arms would otherwise share)
            for (int qx = -1; qx <= 1; qx += 2)
                for (int qz = -1; qz <= 1; qz += 2)
                {
                    Vector3 c = new Vector3(qx * 17.5f, 0, qz * 17.5f);
                    Shapes.Box(root, c + Vector3.up * 0.03f, new Vector3(13f, 0.08f, 13f), M(new Color(0.3f, 0.48f, 0.22f)), default, false);
                    for (int i = 0; i < 4; i++)
                        Tree(c + new Vector3(((float)rnd.NextDouble() - 0.5f) * 10f, 0, ((float)rnd.NextDouble() - 0.5f) * 10f), rnd);
                }
        }

        void Skyline(System.Random rnd)
        {
            Color[] tones = { new Color(0.85f, 0.78f, 0.68f), new Color(0.8f, 0.72f, 0.72f), new Color(0.88f, 0.86f, 0.75f), new Color(0.75f, 0.8f, 0.8f) };
            for (int qx = -1; qx <= 1; qx += 2)
                for (int qz = -1; qz <= 1; qz += 2)
                    for (int i = 0; i < 12; i++)
                    {
                        float x = qx * (45f + (float)rnd.NextDouble() * 100f), z = qz * (45f + (float)rnd.NextDouble() * 100f);
                        int floors = 3 + rnd.Next(10);
                        float h = floors * 3.2f, w = 10f + (float)rnd.NextDouble() * 10f;
                        var fac = Mats.Facade(tones[rnd.Next(tones.Length)], w / 3.2f, floors, rnd.Next(100));
                        facades.Add(fac);
                        Shapes.Box(root, new Vector3(x, h / 2, z), new Vector3(w, h, w), fac);
                        Shapes.Box(root, new Vector3(x, h + 0.15f, z), new Vector3(w + 0.2f, 0.3f, w + 0.2f), Mats.Pbr("concrete", new Vector2(4f, 4f), new Color(0.85f, 0.83f, 0.8f), 0.7f));
                        Shapes.Part(root, PrimitiveType.Cylinder, new Vector3(x + w * 0.25f, h + 1f, z), new Vector3(1.4f, 0.7f, 1.4f), M(new Color(0.06f, 0.06f, 0.06f), 0.3f));
                    }
        }

        void Billboards()
        {
            (Vector3 pos, Vector3 face, string msg, Color col)[] boards =
            {
                (new Vector3(24f, 0, 24f), Vector3.back, "HELMET ON. LIFE ON.", new Color(0.9f, 0.3f, 0.05f)),
                (new Vector3(-24f, 0, -24f), Vector3.forward, "DON'T JUMP THE SIGNAL", new Color(0.1f, 0.35f, 0.7f)),
                (new Vector3(-24f, 0, 24f), Vector3.right, "GIVE WAY TO AMBULANCES", new Color(0.75f, 0.1f, 0.1f)),
            };
            foreach (var b in boards)
            {
                var rot = Quaternion.LookRotation(-b.face).eulerAngles;
                Shapes.Part(root, PrimitiveType.Cylinder, b.pos + Vector3.up * 4f, new Vector3(0.4f, 4f, 0.4f), M(new Color(0.35f, 0.35f, 0.35f), 0.4f));
                Shapes.Box(root, b.pos + Vector3.up * 9.5f, new Vector3(9f, 3.5f, 0.3f), M(b.col, 0.3f), rot);
                Shapes.Text(root, b.msg, b.pos + Vector3.up * 9.5f + b.face * 0.2f, 0.75f, Color.white, Quaternion.LookRotation(-b.face).eulerAngles);
            }
        }

        void Tree(Vector3 p, System.Random rnd)
        {
            float s = 0.8f + (float)rnd.NextDouble() * 0.6f;
            Shapes.Part(root, PrimitiveType.Cylinder, p + Vector3.up * 1.6f * s, new Vector3(0.3f, 1.6f, 0.3f) * s, M(new Color(0.35f, 0.25f, 0.17f)));
            var leaf = M(new[] { new Color(0.2f, 0.42f, 0.17f), new Color(0.25f, 0.48f, 0.2f), new Color(0.17f, 0.36f, 0.15f) }[rnd.Next(3)]);
            var leaf2 = M(new Color(0.14f, 0.3f, 0.12f), 0.15f);
            for (int i = 0; i < 7; i++)
            {
                float a = i * 0.9f + (float)rnd.NextDouble(), rr = i == 0 ? 0f : 1.1f + (float)rnd.NextDouble() * 0.6f;
                Shapes.Part(root, PrimitiveType.Sphere, p + new Vector3(Mathf.Cos(a) * rr, 3.4f + (float)rnd.NextDouble() * 1.8f + (i == 0 ? 0.8f : 0f), Mathf.Sin(a) * rr) * s,
                            new Vector3(1f, 0.8f, 1f) * (1.5f + (float)rnd.NextDouble() * 0.9f) * s, i % 3 == 2 ? leaf2 : leaf);
            }
        }

        // ---------- pedestrians (visual only) ----------
        class Walker { public Transform t, legL, legR; public int arm; public float side, d, speed, dir, phase, lane; public bool standing; }

        void Pedestrians(System.Random rnd)
        {
            Color[] clothes = { new Color(0.85f, 0.15f, 0.35f), new Color(0.95f, 0.65f, 0.1f), new Color(0.2f, 0.4f, 0.75f), new Color(0.93f, 0.93f, 0.9f),
                                new Color(0.3f, 0.6f, 0.4f), new Color(0.55f, 0.2f, 0.6f), new Color(0.25f, 0.25f, 0.28f) };
            Color[] skin = { new Color(0.55f, 0.38f, 0.26f), new Color(0.45f, 0.3f, 0.2f), new Color(0.65f, 0.47f, 0.33f) };
            for (int i = 0; i < 80; i++)
            {
                var w = new Walker { arm = rnd.Next(4), side = rnd.NextDouble() < 0.5 ? -1f : 1f, d = 14f + (float)rnd.NextDouble() * 120f,
                                     speed = 1.0f + (float)rnd.NextDouble() * 0.5f, dir = rnd.NextDouble() < 0.5 ? -1f : 1f,
                                     phase = (float)rnd.NextDouble() * 6f, lane = ((float)rnd.NextDouble() - 0.5f) * 1.6f, standing = i % 6 == 0 };
                w.t = new GameObject("Pedestrian").transform;
                w.t.SetParent(root, false);
                bool saree = i % 3 == 0;
                var cloth = M(clothes[rnd.Next(clothes.Length)]);
                if (saree) Shapes.Part(w.t, PrimitiveType.Cylinder, new Vector3(0, 0.5f, 0), new Vector3(0.42f, 0.5f, 0.42f), cloth);
                else
                {
                    var trouser = M(new Color(0.15f, 0.15f, 0.2f));
                    w.legL = Leg(w.t, -0.1f, trouser); w.legR = Leg(w.t, 0.1f, trouser);
                }
                Shapes.Part(w.t, PrimitiveType.Capsule, new Vector3(0, 1.22f, 0), new Vector3(0.42f, 0.36f, 0.28f), cloth);
                Shapes.Part(w.t, PrimitiveType.Sphere, new Vector3(0, 1.68f, 0), Vector3.one * 0.24f, M(skin[rnd.Next(3)]));
                Shapes.Part(w.t, PrimitiveType.Sphere, new Vector3(0, 1.74f, -0.03f), new Vector3(0.25f, 0.16f, 0.26f), M(new Color(0.05f, 0.04f, 0.03f)), default, false);
                walkers.Add(w);
            }
        }

        static Transform Leg(Transform p, float x, Material m)
        {
            var pivot = new GameObject("Leg").transform;
            pivot.SetParent(p, false); pivot.localPosition = new Vector3(x, 0.92f, 0);
            Shapes.Box(pivot, new Vector3(0, -0.45f, 0), new Vector3(0.15f, 0.9f, 0.17f), m);
            return pivot;
        }

        public void UpdatePedestrians(float dt, float time)
        {
            foreach (var w in walkers)
            {
                if (!w.standing)
                {
                    w.d += w.dir * w.speed * dt;
                    if (w.d > L - 6f || w.d < Half + 6f) w.dir = -w.dir;
                }
                Vector3 u = Geo.U[w.arm], lf = Geo.Left(-u);
                w.t.position = u * w.d + lf * w.side * (Half + Foot / 2 + w.lane * 0.5f) + Vector3.up * 0.16f;
                w.t.rotation = Quaternion.LookRotation(w.standing ? -lf * w.side : u * w.dir);
                if (w.legL != null && !w.standing)
                {
                    float a = Mathf.Sin(time * w.speed * 5f + w.phase) * 25f;
                    w.legL.localRotation = Quaternion.Euler(a, 0, 0);
                    w.legR.localRotation = Quaternion.Euler(-a, 0, 0);
                }
            }
        }
    }
}
