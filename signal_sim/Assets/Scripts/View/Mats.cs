using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>Shared materials (cached, so thousands of parts cost a few dozen draw-call states) and procedural textures.</summary>
    public static class Mats
    {
        static Shader std;
        static readonly Dictionary<string, Material> cache = new Dictionary<string, Material>();
        /// <summary>URP Lit when the project runs the Universal Render Pipeline, else the built-in Standard shader</summary>
        public static bool Urp => UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline != null;
        static Shader Std => std != null ? std : (std = Shader.Find(Urp ? "Universal Render Pipeline/Lit" : "Standard"));
        static string SmoothProp => Urp ? "_Smoothness" : "_Glossiness";

        static void Smoothness(Material m, float v) { m.SetFloat(SmoothProp, v); m.SetFloat("_GlossMapScale", v); }

        /// <summary>smoothness read from the albedo alpha (how the Poly Haven maps are packed)</summary>
        static void SmoothFromAlbedo(Material m, float scale)
        {
            m.SetFloat("_SmoothnessTextureChannel", 1f);
            m.EnableKeyword("_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A");
            Smoothness(m, scale);
        }

        static void Normal(Material m, Texture2D n, float strength)
        {
            if (n == null) return;
            m.SetTexture("_BumpMap", n);
            m.SetFloat("_BumpScale", strength);
            m.EnableKeyword("_NORMALMAP");
        }

        /// <summary>photo-scanned surface from Resources/PH (signal_sim/Tools/fetch_assets.py); plain colour if not fetched</summary>
        public static Material Pbr(string name, Vector2 tiling, Color tint, float smooth = 1f, float bump = 1f)
        {
            string k = $"pbr{name},{tiling.x:F2},{tiling.y:F2},{tint.r:F2},{tint.g:F2},{tint.b:F2},{smooth:F2}";
            if (cache.TryGetValue(k, out var m)) return m;
            var albedo = Resources.Load<Texture2D>($"PH/{name}_albedo");
            if (albedo == null) return cache[k] = Solid(tint * 0.6f, 0.2f);
            m = new Material(Std) { color = tint, mainTexture = albedo, mainTextureScale = tiling };
            Normal(m, Resources.Load<Texture2D>($"PH/{name}_normal"), bump);
            SmoothFromAlbedo(m, smooth);
            return cache[k] = m;
        }

        public static Material Solid(Color c, float gloss = 0.15f, float metal = 0f)
        {
            string k = $"s{c.r:F3},{c.g:F3},{c.b:F3},{gloss:F2},{metal:F2}";
            if (cache.TryGetValue(k, out var m)) return m;
            m = new Material(Std) { color = c };
            Smoothness(m, gloss);
            m.SetFloat("_Metallic", metal);
            return cache[k] = m;
        }

        public static Material Glow(Color c, float intensity)
        {
            string k = $"g{c.r:F3},{c.g:F3},{c.b:F3},{intensity:F2}";
            if (cache.TryGetValue(k, out var m)) return m;
            m = new Material(Std) { color = c };
            m.EnableKeyword("_EMISSION");
            m.SetColor("_EmissionColor", c * intensity);
            m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
            return cache[k] = m;
        }

        public static Material Textured(string key, Texture2D tex, Vector2 tiling, Color tint, float gloss = 0.1f)
        {
            if (cache.TryGetValue(key, out var m)) return m;
            m = new Material(Std) { color = tint, mainTexture = tex, mainTextureScale = tiling };
            Smoothness(m, gloss);
            return cache[key] = m;
        }

        /// <summary>not cached: buildings each get their own facade tiling and night-time window glow</summary>
        public static Material Facade(Color tint, float bays, float floors, int seed)
        {
            var m = new Material(Std) { color = tint, mainTexture = FacadeTex, mainTextureScale = new Vector2(bays, floors) };
            SmoothFromAlbedo(m, 1f);                                         // glossy window glass, matte walls
            Normal(m, Resources.Load<Texture2D>("PH/plaster_normal"), 0.6f);   // plaster relief on the walls
            m.EnableKeyword("_EMISSION");
            m.SetTexture("_EmissionMap", LitWindows(seed % 4));
            m.SetColor("_EmissionColor", Color.black);
            m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
            return m;
        }

        // ---------- textures ----------
        static Texture2D asphalt, paving, facade, kerb;
        static readonly Texture2D[] lit = new Texture2D[4];

        static Texture2D New(int w, int h, FilterMode f = FilterMode.Bilinear) =>
            new Texture2D(w, h, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat, filterMode = f, anisoLevel = 4 };

        public static Texture2D Asphalt
        {
            get
            {
                if (asphalt) return asphalt;
                asphalt = New(256, 256);
                var rnd = new System.Random(3);
                for (int y = 0; y < 256; y++)
                    for (int x = 0; x < 256; x++)
                    {
                        float n = Mathf.PerlinNoise(x * 0.05f, y * 0.05f) * 0.5f + Mathf.PerlinNoise(x * 0.3f, y * 0.3f) * 0.3f + (float)rnd.NextDouble() * 0.2f;
                        float g = 0.78f + n * 0.3f;
                        asphalt.SetPixel(x, y, new Color(g, g, g * 0.98f));
                    }
                asphalt.Apply();
                return asphalt;
            }
        }

        public static Texture2D Paving
        {
            get
            {
                if (paving) return paving;
                paving = New(64, 64);
                var rnd = new System.Random(4);
                for (int y = 0; y < 64; y++)
                    for (int x = 0; x < 64; x++)
                    {
                        bool grout = x % 32 < 2 || y % 32 < 2;
                        float g = grout ? 0.55f : 0.85f + (float)rnd.NextDouble() * 0.12f;
                        paving.SetPixel(x, y, new Color(g, g * 0.95f, g * 0.9f));
                    }
                paving.Apply();
                return paving;
            }
        }

        public static Texture2D Kerb
        {
            get
            {
                if (kerb) return kerb;
                kerb = New(2, 1, FilterMode.Point);
                kerb.SetPixels(new[] { new Color(0.95f, 0.8f, 0.1f), new Color(0.08f, 0.08f, 0.08f) });
                kerb.Apply();
                return kerb;
            }
        }

        /// <summary>one bay x one floor: wall, window with frame and sill</summary>
        public static Texture2D FacadeTex
        {
            get
            {
                if (facade) return facade;
                facade = New(32, 32);
                for (int y = 0; y < 32; y++)
                    for (int x = 0; x < 32; x++)
                    {
                        bool win = x >= 8 && x < 24 && y >= 10 && y < 26;
                        bool frame = !win && x >= 7 && x < 25 && y >= 9 && y < 27;
                        bool sill = y >= 7 && y < 9 && x >= 6 && x < 26;
                        Color c = win ? new Color(0.16f, 0.2f, 0.24f, 0.92f) : frame ? new Color(0.9f, 0.9f, 0.88f, 0.35f)
                                : sill ? new Color(0.6f, 0.6f, 0.6f, 0.2f) : new Color(1f, 1f, 1f, 0.12f);
                        facade.SetPixel(x, y, c);
                    }
                facade.Apply();
                return facade;
            }
        }

        /// <summary>emission map: some windows lit at night (4 variants so streets don't look tiled)</summary>
        static Texture2D LitWindows(int v)
        {
            if (lit[v]) return lit[v];
            var t = New(128, 128);
            var rnd = new System.Random(10 + v);
            var on = new bool[4, 4];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) on[i, j] = rnd.NextDouble() < 0.45;
            for (int y = 0; y < 128; y++)
                for (int x = 0; x < 128; x++)
                {
                    int bx = x / 32, by = y / 32, lx = x % 32, ly = y % 32;
                    bool win = lx >= 8 && lx < 24 && ly >= 10 && ly < 26;
                    t.SetPixel(x, y, win && on[bx, by] ? new Color(1f, 0.85f, 0.55f) : Color.black);
                }
            t.Apply();
            return lit[v] = t;
        }
    }

    public static class Shapes
    {
        public static void Kill(Object o)
        {
            if (o == null) return;
            if (Application.isPlaying) Object.Destroy(o); else Object.DestroyImmediate(o);
        }

        public static Transform Part(Transform parent, PrimitiveType type, Vector3 pos, Vector3 scale, Material m,
                                     Vector3 euler = default, bool shadows = true, string name = null)
        {
            var g = GameObject.CreatePrimitive(type);
            if (name != null) g.name = name;
            Kill(g.GetComponent<Collider>());
            var t = g.transform;
            t.SetParent(parent, false);
            t.localPosition = pos;
            t.localRotation = Quaternion.Euler(euler);
            t.localScale = scale;
            var r = g.GetComponent<Renderer>();
            r.sharedMaterial = m;
            if (!shadows) r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            return t;
        }

        public static Transform Box(Transform parent, Vector3 pos, Vector3 scale, Material m, Vector3 euler = default, bool shadows = true) =>
            Part(parent, PrimitiveType.Cube, pos, scale, m, euler, shadows);

        static Font font;
        static Material textMat;
        public static Font Font => font != null ? font : (font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf"));

        /// <summary>depth-tested text material (Shaders/Text3D.shader), kept pointing at the font atlas when it is rebuilt</summary>
        static Material TextMat
        {
            get
            {
                if (textMat != null) return textMat;
                var sh = Shader.Find("SignalSim/Text3D");
                if (sh == null) return textMat = Font.material;
                textMat = new Material(sh) { mainTexture = Font.material.mainTexture };
                Font.textureRebuilt += f => { if (f == font && textMat != null) textMat.mainTexture = f.material.mainTexture; };
                return textMat;
            }
        }

        /// <summary>world text, readable when looking along the text's +z</summary>
        public static TextMesh Text(Transform parent, string s, Vector3 pos, float height, Color c, Vector3 euler = default, FontStyle style = FontStyle.Bold)
        {
            var g = new GameObject("Text");
            g.transform.SetParent(parent, false);
            g.transform.localPosition = pos;
            g.transform.localRotation = Quaternion.Euler(euler);
            var tm = g.AddComponent<TextMesh>();
            tm.font = Font; tm.text = s; tm.fontSize = 64; tm.characterSize = height / 6.4f * 1.0f; tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center; tm.fontStyle = style; tm.color = c;
            var r = g.GetComponent<MeshRenderer>();
            r.sharedMaterial = TextMat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            return tm;
        }
    }
}
