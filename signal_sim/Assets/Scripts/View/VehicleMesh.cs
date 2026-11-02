using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// Smooth vehicle bodies lofted through cross-sections along the length (z). Each section is a rounded trapezoid
    /// (superellipse): bottom/top height, half-width at bottom/top, and squareness n (2 = oval, 8+ = nearly boxy).
    /// Meshes are cached per shape, so hundreds of vehicles share a handful of meshes.
    /// </summary>
    public static class VehicleMesh
    {
        public readonly struct Sec
        {
            public readonly float z, y0, y1, w0, w1, n;
            public Sec(float z, float y0, float y1, float w0, float w1, float n = 4f) { this.z = z; this.y0 = y0; this.y1 = y1; this.w0 = w0; this.w1 = w1; this.n = n; }
        }

        static readonly Dictionary<string, Mesh> cache = new Dictionary<string, Mesh>();

        public static Mesh Loft(string key, Sec[] s, int ring = 28)
        {
            if (cache.TryGetValue(key, out var mesh)) return mesh;
            var v = new List<Vector3>(); var tri = new List<int>();
            foreach (var sec in s)
                for (int k = 0; k < ring; k++) v.Add(Point(sec, k, ring));
            for (int i = 0; i < s.Length - 1; i++)
                for (int k = 0; k < ring; k++)
                {
                    int a = i * ring + k, b = i * ring + (k + 1) % ring, c = (i + 1) * ring + k, d = (i + 1) * ring + (k + 1) % ring;
                    tri.AddRange(new[] { a, b, c, b, d, c });
                }
            Cap(s[s.Length - 1], ring, v, tri, true);
            Cap(s[0], ring, v, tri, false);
            mesh = new Mesh { name = key };
            mesh.SetVertices(v); mesh.SetTriangles(tri, 0);
            mesh.RecalculateNormals(); mesh.RecalculateBounds();
            return cache[key] = mesh;
        }

        static Vector3 Point(Sec s, int k, int ring)
        {
            float th = -Mathf.PI / 2f + 2f * Mathf.PI * k / ring;   // start at the bottom so the seam is hidden underneath
            float c = Mathf.Cos(th), sn = Mathf.Sin(th);
            float px = Mathf.Sign(c) * Mathf.Pow(Mathf.Abs(c), 2f / s.n), py = Mathf.Sign(sn) * Mathf.Pow(Mathf.Abs(sn), 2f / s.n);
            float t = (py + 1f) * 0.5f;
            return new Vector3(px * Mathf.Lerp(s.w0, s.w1, t), Mathf.Lerp(s.y0, s.y1, t), s.z);
        }

        /// <summary>flat end cap with its own vertices (crisp edge)</summary>
        static void Cap(Sec s, int ring, List<Vector3> v, List<int> tri, bool front)
        {
            int c0 = v.Count;
            v.Add(new Vector3(0f, (s.y0 + s.y1) * 0.5f, s.z));
            for (int k = 0; k < ring; k++) v.Add(Point(s, k, ring));
            for (int k = 0; k < ring; k++)
            {
                int a = c0 + 1 + k, b = c0 + 1 + (k + 1) % ring;
                if (front) tri.AddRange(new[] { c0, a, b }); else tri.AddRange(new[] { c0, b, a });
            }
        }

        public static Transform Part(Transform parent, Mesh mesh, Material m, string name = "Body")
        {
            var g = new GameObject(name);
            g.transform.SetParent(parent, false);
            g.AddComponent<MeshFilter>().sharedMesh = mesh;
            g.AddComponent<MeshRenderer>().sharedMaterial = m;
            return g.transform;
        }

        // ---------- shapes ----------
        public static Mesh CarBody(float L, float W)
        {
            float h = W / 2f;
            return Loft($"car{L:F2}{W:F2}", new[]
            {
                new Sec(-L / 2, 0.42f, 0.80f, h * 0.80f, h * 0.74f, 3f),
                new Sec(-L / 2 + 0.12f, 0.32f, 0.92f, h * 0.95f, h * 0.90f, 3.5f),
                new Sec(-L * 0.32f, 0.30f, 0.98f, h, h * 0.94f, 4f),
                new Sec(L * 0.18f, 0.30f, 0.98f, h, h * 0.94f, 4f),
                new Sec(L * 0.30f, 0.30f, 0.90f, h * 0.99f, h * 0.92f, 4f),
                new Sec(L / 2 - 0.14f, 0.32f, 0.76f, h * 0.95f, h * 0.86f, 3.5f),
                new Sec(L / 2, 0.40f, 0.66f, h * 0.82f, h * 0.76f, 3f),
            });
        }

        public static Mesh CarGlass(float L, float W)
        {
            float h = W / 2f;
            return Loft($"carglass{L:F2}{W:F2}", new[]
            {
                new Sec(-L * 0.36f, 0.95f, 0.99f, h * 0.88f, h * 0.86f),
                new Sec(-L * 0.24f, 0.95f, 1.40f, h * 0.90f, h * 0.72f),
                new Sec(L * 0.02f, 0.95f, 1.43f, h * 0.90f, h * 0.73f),
                new Sec(L * 0.08f, 0.95f, 1.40f, h * 0.90f, h * 0.73f),
                new Sec(L * 0.22f, 0.96f, 1.00f, h * 0.89f, h * 0.86f),
            });
        }

        public static Mesh CarRoof(float L, float W)
        {
            float h = W / 2f;
            return Loft($"carroof{L:F2}{W:F2}", new[]
            {
                new Sec(-L * 0.23f, 1.37f, 1.43f, h * 0.73f, h * 0.70f, 6f),
                new Sec(-L * 0.20f, 1.38f, 1.455f, h * 0.745f, h * 0.715f, 6f),
                new Sec(L * 0.05f, 1.39f, 1.465f, h * 0.75f, h * 0.72f, 6f),
                new Sec(L * 0.08f, 1.37f, 1.44f, h * 0.74f, h * 0.71f, 6f),
            });
        }

        public static Mesh AutoBody() => Loft("autobody", new[]
        {
            new Sec(-1.36f, 0.36f, 0.95f, 0.69f, 0.67f, 3f),
            new Sec(-1.25f, 0.33f, 1.0f, 0.70f, 0.68f, 4f),
            new Sec(0.45f, 0.33f, 1.0f, 0.70f, 0.66f, 4f),
            new Sec(0.80f, 0.36f, 0.98f, 0.52f, 0.42f, 3f),
            new Sec(1.15f, 0.40f, 0.92f, 0.34f, 0.27f, 2.5f),
            new Sec(1.36f, 0.48f, 0.84f, 0.2f, 0.16f, 2.2f),
        });

        public static Mesh AutoCanopy() => Loft("autocanopy", new[]
        {
            new Sec(-1.38f, 1.58f, 1.80f, 0.66f, 0.60f, 3f),
            new Sec(-1.22f, 1.66f, 1.87f, 0.715f, 0.69f, 5f),
            new Sec(0.55f, 1.66f, 1.87f, 0.715f, 0.69f, 5f),
            new Sec(0.92f, 1.56f, 1.80f, 0.62f, 0.50f, 3f),
        });

        public static Mesh BusPart(float L, float y0, float y1, string tag) => Loft($"bus{tag}{L:F1}", new[]
        {
            new Sec(-L / 2, y0 + 0.05f, y1 - 0.05f, 1.2f, 1.18f, 6f),
            new Sec(-L / 2 + 0.25f, y0, y1, 1.25f, 1.23f, 9f),
            new Sec(L / 2 - 0.25f, y0, y1, 1.25f, 1.23f, 9f),
            new Sec(L / 2, y0 + 0.05f, y1 - 0.05f, 1.2f, 1.17f, 6f),
        });

        public static Mesh VanBody(float L) => Loft($"van{L:F1}", new[]
        {
            new Sec(-L / 2, 0.42f, 2.25f, 0.97f, 0.93f, 6f),
            new Sec(-L / 2 + 0.15f, 0.36f, 2.32f, 1.0f, 0.96f, 8f),
            new Sec(L / 2 - 1.1f, 0.36f, 2.32f, 1.0f, 0.96f, 8f),
            new Sec(L / 2 - 0.55f, 0.36f, 1.65f, 0.99f, 0.93f, 6f),
            new Sec(L / 2 - 0.12f, 0.40f, 1.20f, 0.97f, 0.9f, 5f),
            new Sec(L / 2, 0.46f, 1.0f, 0.9f, 0.84f, 4f),
        });
    }
}
