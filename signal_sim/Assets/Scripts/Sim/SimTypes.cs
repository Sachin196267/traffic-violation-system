using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    public enum VType { Bike, Bicycle, Auto, Car, Bus, Truck, Ambulance, Cattle }
    public enum Turn { Left, Straight, Right }
    public enum Light { Red, Amber, Green }

    public readonly struct VSpec
    {
        public readonly float len, wid, height, v0, a, b, pcu, persons;
        public readonly Color col;

        public VSpec(float len, float wid, float height, float v0, float a, float b, float pcu, float persons, Color col)
        {
            this.len = len; this.wid = wid; this.height = height; this.v0 = v0; this.a = a; this.b = b;
            this.pcu = pcu; this.persons = persons; this.col = col;
        }
    }

    public static class Specs
    {
        // pcu: IRC:106-1990 urban values (two-wheeler 0.5, auto 1.2, bus/truck 2.2, bicycle 0.4)
        // persons: typical Indian urban occupancy (bus ~40 standing + seated)
        public static readonly Dictionary<VType, VSpec> All = new Dictionary<VType, VSpec>
        {
            { VType.Bike,      new VSpec(2.0f, 0.8f, 1.4f, 13f,  2.5f, 3f, 0.5f, 1.4f, new Color(0.15f, 0.15f, 0.18f)) },
            { VType.Bicycle,   new VSpec(1.8f, 0.6f, 1.4f, 4.5f, 0.8f, 2f, 0.4f, 1.0f, new Color(0.25f, 0.35f, 0.6f)) },
            { VType.Auto,      new VSpec(2.8f, 1.4f, 1.7f, 10f,  1.5f, 3f, 1.2f, 2.5f, new Color(0.15f, 0.55f, 0.2f)) },
            { VType.Car,       new VSpec(4.2f, 1.8f, 1.5f, 14f,  2.0f, 3f, 1.0f, 1.8f, new Color(0.75f, 0.75f, 0.78f)) },
            { VType.Bus,       new VSpec(11f,  2.5f, 3.0f, 11f,  1.0f, 3f, 2.2f, 40f,  new Color(0.8f, 0.2f, 0.15f)) },
            { VType.Truck,     new VSpec(8.5f, 2.5f, 3.0f, 10f,  0.9f, 3f, 2.2f, 1.5f, new Color(0.9f, 0.6f, 0.1f)) },
            { VType.Ambulance, new VSpec(5.0f, 2.0f, 2.3f, 16f,  2.5f, 3f, 1.0f, 3f,   Color.white) },
            { VType.Cattle,    new VSpec(2.2f, 0.9f, 1.4f, 0f,   0f,   3f, 0f,   0f,   new Color(0.85f, 0.82f, 0.75f)) },
        };

        /// <summary>arrival mix of an Indian city arterial</summary>
        public static readonly (VType type, float share)[] Mix =
        {
            (VType.Bike, 0.38f), (VType.Bicycle, 0.04f), (VType.Car, 0.28f), (VType.Auto, 0.16f), (VType.Bus, 0.06f), (VType.Truck, 0.08f),
        };

        public static float MeanPcu()
        {
            float s = 0f;
            foreach (var m in Mix) s += m.share * All[m.type].pcu;
            return s;
        }
    }

    /// <summary>Polyline path with arc-length lookup.</summary>
    public class Path
    {
        readonly List<Vector3> pts;
        readonly float[] cum;
        public float Length => cum[cum.Length - 1];

        public Path(List<Vector3> pts)
        {
            this.pts = pts;
            cum = new float[pts.Count];
            for (int i = 1; i < pts.Count; i++) cum[i] = cum[i - 1] + Vector3.Distance(pts[i - 1], pts[i]);
        }

        public void Eval(float s, out Vector3 pos, out Vector3 dir)
        {
            s = Mathf.Clamp(s, 0f, Length);
            int lo = 0, hi = cum.Length - 1;
            while (hi - lo > 1) { int m = (lo + hi) / 2; if (cum[m] <= s) lo = m; else hi = m; }
            float seg = cum[hi] - cum[lo];
            dir = (pts[hi] - pts[lo]).normalized;
            pos = seg > 1e-5f ? Vector3.Lerp(pts[lo], pts[hi], (s - cum[lo]) / seg) : pts[lo];
        }
    }

    public class Vehicle
    {
        public int id;
        public VType type;
        public VSpec spec;
        public int approach, lane, exitApproach, exitLane;
        public Turn turn;
        public Path path;
        public float s, v, acc, sStop;     // s = front-bumper arc length along the path
        public float lateral, targetLateral;
        public float spawnT, crossT = -1f, wait;
        public float patience = float.MaxValue;   // red wait after which this driver jumps the signal
        public float stopOffset;                  // two-wheelers creep past the stop line onto the zebra
        public bool jumper, inZone;
        public float dwellS = -1f, dwellLeft;     // bus stop / auto pickup just past the junction
        public float life;                        // cattle: seconds left standing
        public Vector3 pos, dir;
        public bool Crossed => crossT >= 0f;
        public bool Obstacle => type == VType.Cattle;
        public float DistToStop => sStop - s;
        public float SCross => sStop + 3f;        // past the zebra = through the signal
    }

    /// <summary>
    /// Four-arm junction, left-hand traffic, two lanes per direction. x = east, z = north.
    /// Lane 0 = inner (next to the centre line, right turns), lane 1 = kerb (left turns, bus stops, cattle).
    /// </summary>
    public static class Geo
    {
        public const float W = 3.5f;    // lane width
        public const float J = 9f;      // stop line distance from the junction centre
        public const float L = 150f;    // arm length
        public const float Box = 2 * W; // half-size of the junction box
        public const int Inner = 0, Kerb = 1;
        public static readonly string[] Names = { "North", "East", "South", "West" };
        public static readonly Vector3[] U = { Vector3.forward, Vector3.right, Vector3.back, Vector3.left };

        public static Vector3 Left(Vector3 d) => new Vector3(-d.z, 0f, d.x);

        public static int ApproachOf(Vector3 u)
        {
            for (int i = 0; i < 4; i++) if (Vector3.Dot(U[i], u) > 0.9f) return i;
            return 0;
        }

        /// <summary>exit arm for a movement: Indian roads drive on the left, so the left turn is the short one</summary>
        public static int ExitOf(int a, Turn t)
        {
            Vector3 d = -U[a];
            Vector3 ub = t == Turn.Straight ? d : t == Turn.Left ? Left(d) : -Left(d);
            return ApproachOf(ub);
        }

        public static bool InBox(Vector3 p) => Mathf.Abs(p.x) < Box && Mathf.Abs(p.z) < Box;

        public static Path BuildPath(int a, int lane, int b, int exitLane, out float sStop)
        {
            Vector3 d = -U[a], ub = U[b];
            Vector3 start = U[a] * L + Left(d) * (lane + 0.5f) * W;
            Vector3 p0 = U[a] * J + Left(d) * (lane + 0.5f) * W;
            Vector3 p2 = ub * J + Left(ub) * (exitLane + 0.5f) * W;
            Vector3 end = ub * L + Left(ub) * (exitLane + 0.5f) * W;
            Vector3 c = Mathf.Abs(Vector3.Dot(d, ub)) > 0.9f ? (p0 + p2) * 0.5f : p0 + d * Vector3.Dot(p2 - p0, d);
            var pts = new List<Vector3> { start };
            for (int i = 0; i <= 16; i++)
            {
                float t = i / 16f;
                pts.Add((1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * c + t * t * p2);
            }
            pts.Add(end);
            sStop = L - J;
            return new Path(pts);
        }
    }
}
