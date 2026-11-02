using System;
using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>What a junction camera (detection range) reports per approach. Controllers read only this and Visible().</summary>
    public struct ApproachState
    {
        public int queued;            // stopped/slow signal-controlled vehicles in camera range
        public float pcu, persons;    // their PCU and estimated people
        public float maxWait;         // longest wait among them, s
        public int near;              // moving vehicles within 40 m of the stop line
        public bool ambulance;        // ambulance in ambulance-detection range, not yet through
        public float ambDist, ambSpeed;
        public int crossed;           // since the stats reset (dashboard)
        public float crossedPcuTotal, arrivedPcuTotal;   // cumulative counters (for learning controllers)
    }

    public class Stats
    {
        public int served, ambServed, violations;
        public double waitSum, personWaitSum, persons, ambWaitSum;
        public float maxWait, boxBlocked, since;
        public float AvgWait => served > 0 ? (float)(waitSum / served) : 0f;
        public float PersonWait => persons > 0 ? (float)(personWaitSum / persons) : 0f;
        public float AmbAvgWait => ambServed > 0 ? (float)(ambWaitSum / ambServed) : 0f;
    }

    public class SimOptions
    {
        public bool freeLeft = true;       // left turn not held by the signal (slip lane)
        public bool redJumping = true;     // impatient riders/drivers jump the red
        public bool busStops = true;       // buses and autos stop just past the junction
        public float cattlePerHour = 0f;   // stray cattle standing in the kerb lane
        public float detectRange = 100f;   // camera queue detection, m before the stop line
        public float ambulanceRange = 100f;
        public float[] mix;                // optional arrival shares in Specs.Mix order (e.g. measured from a video)
        public float patienceScale = 1f;   // sensitivity tests: how long drivers wait before jumping a red
        public float dwellScale = 1f;      // sensitivity tests: bus / auto stop duration

        public float Share(int i)
        {
            if (mix == null || mix.Length != Specs.Mix.Length) return Specs.Mix[i].share;
            float sum = 0f; foreach (var m in mix) sum += m;
            return sum > 0f ? mix[i] / sum : Specs.Mix[i].share;
        }

        public float MeanPcu()
        {
            float s = 0f;
            for (int i = 0; i < Specs.Mix.Length; i++) s += Share(i) * Specs.All[Specs.Mix[i].type].pcu;
            return s;
        }
    }

    /// <summary>
    /// Pure-C# microscopic simulation of one Indian four-arm junction (no UnityEngine objects, so it runs headless).
    /// Car following = Intelligent Driver Model; lane-less two-wheelers; red-light jumping; free left; bus-stop spillback.
    /// </summary>
    public class SimCore
    {
        public const float DT = 0.1f;
        public const float AmberSec = 3f, AllRedSec = 2f;

        public float t;
        public readonly SimOptions opt;
        public readonly float[] rate = { 700, 500, 700, 500 };   // arrivals, veh/h per approach
        public float ambulancePerHour = 4f;
        public readonly List<Vehicle> vehicles = new List<Vehicle>();
        public readonly List<Vehicle>[] pending = new List<Vehicle>[8];   // generated, no room yet to enter (per lane)
        public readonly ApproachState[] app = new ApproachState[4];
        public readonly float[] exitOcc = new float[4];    // share of the first 40 m of each exit arm that is occupied
        public int boxBlockers;                             // bit per approach with a vehicle stuck in the junction box
        public readonly Stats stats = new Stats();
        public ISignalController controller;

        // signal plant: one approach green at a time (split phasing, the common Indian practice)
        public int green;
        public enum Phase { Green, Amber, AllRed }
        public Phase phase = Phase.Green;
        public float phaseT, greenDur = 20f;
        public string reason = "start";

        readonly System.Random[] arrRng = new System.Random[4];
        readonly System.Random ambRng, cattleRng, misc;
        readonly float[] nextArr = new float[4];
        float nextAmb, nextCattle;
        int nextId;
        readonly List<Vehicle>[] byApp = new List<Vehicle>[4], byExit = new List<Vehicle>[4];

        public SimCore(int seed, ISignalController controller, SimOptions opt = null)
        {
            this.controller = controller;
            this.opt = opt ?? new SimOptions();
            for (int i = 0; i < 8; i++) pending[i] = new List<Vehicle>();
            for (int i = 0; i < 4; i++) { byApp[i] = new List<Vehicle>(); byExit[i] = new List<Vehicle>(); }
            // separate random streams so every controller sees exactly the same arrivals
            for (int a = 0; a < 4; a++) { arrRng[a] = new System.Random(seed * 31 + a); nextArr[a] = Exp(arrRng[a], rate[a]); }
            ambRng = new System.Random(seed * 31 + 7);
            cattleRng = new System.Random(seed * 31 + 8);
            misc = new System.Random(seed * 31 + 9);
            nextAmb = Exp(ambRng, ambulancePerHour);
            nextCattle = Exp(cattleRng, this.opt.cattlePerHour);
            Observe();
            controller.Next(this, out green, out greenDur, out reason);
        }

        static float Exp(System.Random r, float perHour) =>
            perHour <= 0 ? float.MaxValue : (float)(-Math.Log(1 - r.NextDouble()) * 3600.0 / perHour);

        public Light LightFor(int a)
        {
            if (a != green || phase == Phase.AllRed) return Light.Red;
            return phase == Phase.Green ? Light.Green : Light.Amber;
        }

        public bool Controlled(Vehicle v) => !v.Obstacle && !(opt.freeLeft && v.turn == Turn.Left);

        /// <summary>a vehicle the junction camera can see waiting for the signal</summary>
        public bool Visible(Vehicle v) => !v.Crossed && Controlled(v) && v.DistToStop < opt.detectRange;

        public void ResetStats()
        {
            stats.served = stats.ambServed = stats.violations = 0;
            stats.waitSum = stats.personWaitSum = stats.persons = stats.ambWaitSum = 0;
            stats.maxWait = stats.boxBlocked = 0; stats.since = t;
            for (int a = 0; a < 4; a++) app[a].crossed = 0;
        }

        public void SpawnAmbulance(int a) => Generate(a, VType.Ambulance, Turn.Straight, misc);

        public void Step()
        {
            t += DT;
            Arrivals();
            Insert();
            Observe();
            Signal();
            Move();
        }

        // ---------------- arrivals ----------------
        void Arrivals()
        {
            for (int a = 0; a < 4; a++)
            {
                nextArr[a] -= DT;
                while (nextArr[a] <= 0)
                {
                    var r = arrRng[a];
                    double u = r.NextDouble(), w = r.NextDouble();
                    VType ty = Specs.Mix[Specs.Mix.Length - 1].type;
                    double c = 0;
                    for (int i = 0; i < Specs.Mix.Length; i++) { c += opt.Share(i); if (u < c) { ty = Specs.Mix[i].type; break; } }
                    Turn tu = w < 0.25 ? Turn.Left : w < 0.80 ? Turn.Straight : Turn.Right;
                    Generate(a, ty, tu, r);
                    nextArr[a] += Exp(r, rate[a]);
                }
            }
            nextAmb -= DT;
            while (nextAmb <= 0)
            {
                Generate(ambRng.Next(4), VType.Ambulance, Turn.Straight, ambRng);
                nextAmb += Exp(ambRng, ambulancePerHour);
            }
            nextCattle -= DT;
            while (nextCattle <= 0)
            {
                PlaceCattle(cattleRng);
                nextCattle += Exp(cattleRng, opt.cattlePerHour);
            }
        }

        void Generate(int a, VType ty, Turn tu, System.Random r)
        {
            // the same number of random draws whatever the state, so arrivals stay identical across controllers
            double r1 = r.NextDouble(), r2 = r.NextDouble(), r3 = r.NextDouble(), r4 = r.NextDouble(), r5 = r.NextDouble();
            int lane = tu == Turn.Left ? Geo.Kerb : tu == Turn.Right ? Geo.Inner : LaneLoad(a, 0) <= LaneLoad(a, 1) ? 0 : 1;
            if (ty == VType.Bus && tu == Turn.Straight) lane = Geo.Kerb;   // buses keep to the kerb for the bus stop
            int b = Geo.ExitOf(a, tu);
            var sp = Specs.All[ty];
            var v = new Vehicle { id = nextId++, type = ty, spec = sp, approach = a, lane = lane, turn = tu, exitApproach = b, exitLane = lane, spawnT = t };
            v.path = Geo.BuildPath(a, lane, b, lane, out v.sStop);
            bool two = ty == VType.Bike || ty == VType.Bicycle;
            v.lateral = v.targetLateral = two ? (float)(r1 * 2.8 - 1.4) : ty == VType.Auto ? (float)(r1 * 1.2 - 0.6) : 0f;
            v.stopOffset = two ? (float)(r2 * 2.5) : ty == VType.Auto ? (float)r2 : 0f;
            v.patience = ty == VType.Bike ? 40f + (float)r3 * 110f : ty == VType.Bicycle ? 30f + (float)r3 * 90f
                       : ty == VType.Auto ? 60f + (float)r3 * 140f : ty == VType.Car ? 120f + (float)r3 * 300f : float.MaxValue;
            if (v.patience < float.MaxValue) v.patience *= opt.patienceScale;
            if (opt.busStops && ((ty == VType.Bus && r4 < 0.7) || (ty == VType.Auto && r4 < 0.3)))
            {
                v.dwellS = v.path.Length - (Geo.L - Geo.J) + 20f;   // 29 m past the junction centre
                v.dwellLeft = (ty == VType.Bus ? 10f + (float)r5 * 15f : 4f + (float)r5 * 8f) * opt.dwellScale;
            }
            pending[a * 2 + lane].Add(v);
        }

        void PlaceCattle(System.Random r)
        {
            int a = r.Next(4);
            var v = new Vehicle { id = nextId++, type = VType.Cattle, spec = Specs.All[VType.Cattle], approach = a, lane = Geo.Kerb, turn = Turn.Straight };
            v.exitApproach = Geo.ExitOf(a, Turn.Straight); v.exitLane = Geo.Kerb;
            v.path = Geo.BuildPath(a, Geo.Kerb, v.exitApproach, Geo.Kerb, out v.sStop);
            v.s = v.sStop - (35f + (float)r.NextDouble() * 65f);
            v.lateral = v.targetLateral = 0.9f;
            v.life = 30f + (float)r.NextDouble() * 60f;
            Place(v);
            vehicles.Add(v);
        }

        int LaneLoad(int a, int lane)
        {
            int n = pending[a * 2 + lane].Count;
            foreach (var v in vehicles) if (v.approach == a && v.lane == lane && !v.Crossed && !v.Obstacle) n++;
            return n;
        }

        void Insert()
        {
            for (int g = 0; g < 8; g++)
            {
                var q = pending[g];
                foreach (var p in q) p.wait += DT;   // stuck beyond the modelled arm still counts as waiting
                while (q.Count > 0)
                {
                    var nv = q[0];
                    Vehicle last = null;
                    foreach (var v in vehicles)
                        if (v.approach == g / 2 && v.lane == g % 2 && !v.Crossed && (last == null || v.s < last.s)) last = v;
                    float room = last == null ? float.MaxValue : last.s - last.spec.len;
                    if (room < nv.spec.len + 2f) break;
                    nv.s = nv.spec.len;
                    nv.v = last == null ? nv.spec.v0 * 0.8f : Mathf.Min(nv.spec.v0 * 0.8f, last.v);
                    q.RemoveAt(0);
                    Place(nv);
                    vehicles.Add(nv);
                }
            }
        }

        static void Place(Vehicle v)
        {
            v.path.Eval(v.s - v.spec.len * 0.5f, out var p, out var d);
            v.dir = d;
            v.pos = p + Geo.Left(d) * v.lateral;
        }

        // ---------------- camera ----------------
        void Observe()
        {
            for (int a = 0; a < 4; a++)
            {
                var o = app[a];
                app[a] = new ApproachState { crossed = o.crossed, crossedPcuTotal = o.crossedPcuTotal, arrivedPcuTotal = o.arrivedPcuTotal, ambDist = float.MaxValue };
                exitOcc[a] = 0f;
            }
            boxBlockers = 0;
            foreach (var v in vehicles)
            {
                if (v.Obstacle) continue;
                if (!v.Crossed)
                {
                    ref var s = ref app[v.approach];
                    if (!v.inZone && v.DistToStop < opt.detectRange) { v.inZone = true; s.arrivedPcuTotal += v.spec.pcu; }
                    if (v.type == VType.Ambulance && v.DistToStop < opt.ambulanceRange)
                    {
                        s.ambulance = true;
                        if (v.DistToStop < s.ambDist) { s.ambDist = Mathf.Max(0f, v.DistToStop); s.ambSpeed = v.v; }
                    }
                    if (!Visible(v)) continue;
                    if (v.v < 2f) { s.queued++; s.pcu += v.spec.pcu; s.persons += v.spec.persons; s.maxWait = Mathf.Max(s.maxWait, v.wait); }
                    else if (v.DistToStop < 40f) s.near++;
                }
                else
                {
                    if (Geo.InBox(v.pos) && v.v < 1f) boxBlockers |= 1 << v.approach;
                    float x = Vector3.Dot(v.pos, Geo.U[v.exitApproach]);
                    if (x > Geo.Box && x < Geo.Box + 40f) exitOcc[v.exitApproach] += v.spec.len * v.spec.wid;
                }
            }
            for (int b = 0; b < 4; b++) exitOcc[b] = Mathf.Min(1f, exitOcc[b] / (40f * 2f * Geo.W));
            if (boxBlockers != 0) stats.boxBlocked += DT;
        }

        // ---------------- signal ----------------
        void Signal()
        {
            phaseT += DT;
            switch (phase)
            {
                case Phase.Green:
                    if (!controller.Hold(this, out var why)) { phase = Phase.Amber; phaseT = 0; reason = why; }
                    break;
                case Phase.Amber:
                    if (phaseT >= AmberSec) { phase = Phase.AllRed; phaseT = 0; }
                    break;
                case Phase.AllRed:
                    if (phaseT >= AllRedSec)
                    {
                        controller.Next(this, out green, out greenDur, out reason);
                        phase = Phase.Green; phaseT = 0;
                    }
                    break;
            }
        }

        // ---------------- vehicles ----------------
        void Move()
        {
            for (int g = 0; g < 4; g++) { byApp[g].Clear(); byExit[g].Clear(); }
            foreach (var v in vehicles) { byApp[v.approach].Add(v); byExit[v.exitApproach].Add(v); }

            foreach (var v in vehicles)
            {
                if (v.Obstacle) continue;
                var sp = v.spec;
                float gap = float.MaxValue, dv = 0f;
                Vehicle lead = null;
                Lead(v, byApp[v.approach], ref gap, ref dv, ref lead);
                Lead(v, byExit[v.exitApproach], ref gap, ref dv, ref lead);

                if (!v.Crossed && Controlled(v))
                {
                    var lt = LightFor(v.approach);
                    float stopAt = v.DistToStop + v.stopOffset;
                    if (opt.redJumping && !v.jumper && lt == Light.Red && v.wait > v.patience && stopAt < 4f) v.jumper = true;
                    if (!v.jumper)
                    {
                        bool mustStop = lt == Light.Red || (lt == Light.Amber && stopAt > v.v * v.v / (2f * 4f));
                        if (mustStop && stopAt < gap) { gap = stopAt; dv = v.v; lead = null; }
                    }
                }
                // a vehicle stuck in the box blocks everyone else from entering it (junction gridlock)
                if (!v.Crossed && (boxBlockers & ~(1 << v.approach)) != 0 && v.turn != Turn.Left)
                {
                    float toBox = v.DistToStop + (Geo.J - Geo.Box) - 0.5f;
                    if (toBox > 0f && toBox < gap) { gap = toBox; dv = v.v; lead = null; }
                }
                // bus stop / auto pickup
                if (v.dwellS > 0f)
                {
                    float gd = v.dwellS - v.s;
                    if (gd < gap) { gap = gd; dv = v.v; lead = null; }
                    if (gd < 2.5f && v.v < 0.3f && (v.dwellLeft -= DT) <= 0f) v.dwellS = -1f;
                }
                // cars, autos, buses steer round cattle or a halted bus into the other lane
                bool blocker = lead != null && (lead.Obstacle || (lead.dwellS > 0f && lead.v < 0.3f));
                if (blocker && gap < 10f && v.v < 2f && v.type != VType.Bike && v.type != VType.Bicycle && Mathf.Approximately(v.lateral, v.targetLateral))
                    v.targetLateral = v.lateral + (Vector3.Dot(lead.pos - v.pos, Geo.Left(v.dir)) > 0f ? -Geo.W : Geo.W);
                v.lateral = Mathf.MoveTowards(v.lateral, v.targetLateral, 1.2f * DT);
                v.acc = Idm(v.v, gap, dv, sp, v.type == VType.Bike || v.type == VType.Bicycle);
            }

            for (int i = vehicles.Count - 1; i >= 0; i--)
            {
                var v = vehicles[i];
                if (v.Obstacle)
                {
                    if ((v.life -= DT) <= 0f) vehicles.RemoveAt(i);
                    continue;
                }
                v.v = Mathf.Max(0f, v.v + v.acc * DT);
                v.s += v.v * DT;
                if (!v.Crossed)
                {
                    if (v.v < 0.5f) v.wait += DT;
                    if (v.s >= v.SCross) Cross(v);
                }
                if (v.s - v.spec.len >= v.path.Length) { vehicles.RemoveAt(i); continue; }
                Place(v);
            }
        }

        void Cross(Vehicle v)
        {
            v.crossT = t;
            ref var s = ref app[v.approach];
            s.crossed++; s.crossedPcuTotal += v.spec.pcu;
            stats.served++; stats.waitSum += v.wait; stats.maxWait = Mathf.Max(stats.maxWait, v.wait);
            stats.persons += v.spec.persons; stats.personWaitSum += v.wait * v.spec.persons;
            if (v.type == VType.Ambulance) { stats.ambServed++; stats.ambWaitSum += v.wait; }
            if (v.jumper && Controlled(v) && LightFor(v.approach) == Light.Red) stats.violations++;
        }

        /// <summary>nearest vehicle ahead in roughly the same direction and overlapping laterally</summary>
        static void Lead(Vehicle v, List<Vehicle> cands, ref float gap, ref float dv, ref Vehicle lead)
        {
            Vector3 left = Geo.Left(v.dir);
            foreach (var u in cands)
            {
                if (u == v) continue;
                Vector3 r = u.pos - v.pos;
                float along = Vector3.Dot(r, v.dir);
                if (along <= 0f || along > 80f) continue;
                if (Mathf.Abs(Vector3.Dot(r, left)) > (u.spec.wid + v.spec.wid) * 0.5f + 0.3f) continue;
                if (!u.Obstacle && Vector3.Dot(u.dir, v.dir) < 0.3f) continue;
                // side by side where each sees the other ahead (merging paths): the older vehicle goes first,
                // otherwise both brake for each other forever
                if (!u.Obstacle && Vector3.Dot(-r, u.dir) > 0f && u.id > v.id) continue;
                float g = along - (u.spec.len + v.spec.len) * 0.5f;
                if (g < gap) { gap = g; dv = v.v - u.v; lead = u; }
            }
        }

        static float Idm(float vel, float gap, float dv, VSpec sp, bool twoWheeler)
        {
            float s0 = twoWheeler ? 0.6f : 1.5f, T = twoWheeler ? 0.7f : 1.2f;
            float free = 1f - Mathf.Pow(vel / sp.v0, 4f);
            if (gap == float.MaxValue) return sp.a * free;
            float sStar = s0 + Mathf.Max(0f, vel * T + vel * dv / (2f * Mathf.Sqrt(sp.a * sp.b)));
            float g = Mathf.Max(gap, 0.1f);
            return Mathf.Max(-9f, sp.a * (free - (sStar / g) * (sStar / g)));
        }

        /// <summary>vehicles not yet through the signal (incl. those queued beyond the arm) and their mean wait so far</summary>
        public void Backlog(out int n, out float meanWait)
        {
            n = 0; double w = 0;
            foreach (var v in vehicles) if (!v.Crossed && !v.Obstacle) { n++; w += v.wait; }
            foreach (var q in pending) foreach (var p in q) { n++; w += p.wait; }
            meanWait = n > 0 ? (float)(w / n) : 0f;
        }
    }
}
