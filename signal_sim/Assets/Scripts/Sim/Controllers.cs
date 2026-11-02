using UnityEngine;

namespace SignalSim
{
    public interface ISignalController
    {
        string Name { get; }
        /// <summary>pick the next approach to turn green and its planned green time</summary>
        void Next(SimCore sim, out int approach, out float greenSec, out string reason);
        /// <summary>called every step during green; false ends the green (amber follows)</summary>
        bool Hold(SimCore sim, out string reason);
    }

    public static class Controllers
    {
        public static readonly string[] Names = { "Fixed 30 s", "Webster", "Adaptive v1", "PRAVAH", "Python bridge" };
        public const int Bridge = 4;

        public static ISignalController Make(int i) => i switch
        {
            0 => new FixedTimeController(),
            1 => new WebsterController(),
            2 => new SmartController(),
            3 => new PravahController(),
            _ => new PythonBridgeController(),
        };

        /// <summary>IRC:SP:41 saturation flow S = 525 x approach width (m) PCU/h, here for a 7 m approach</summary>
        public const float IrcSatFlowPcuPerSec = 525f * 2f * Geo.W / 3600f;
    }

    /// <summary>Naive baseline: N -> E -> S -> W, 30 s each, whatever the traffic.</summary>
    public class FixedTimeController : ISignalController
    {
        public float greenSec = 30f;
        public string Name => "Fixed 30 s";

        public void Next(SimCore sim, out int approach, out float g, out string reason)
        {
            approach = sim.t < SimCore.DT * 2 ? 0 : (sim.green + 1) % 4;
            g = greenSec;
            reason = $"fixed cycle: {Geo.Names[approach]} gets {greenSec:F0} s";
        }

        public bool Hold(SimCore sim, out string reason)
        {
            reason = "fixed green time over";
            return sim.phaseT < sim.greenDur;
        }
    }

    /// <summary>
    /// Properly designed fixed-time plan (Webster's method, as used in IRC:SP:41) from the arrival rates a traffic
    /// engineer would have counted: C0 = (1.5 L + 5) / (1 - Y), greens split in proportion to flow ratios y.
    /// </summary>
    public class WebsterController : ISignalController
    {
        public string Name => "Webster";
        readonly float[] g = new float[4];
        float cycle;

        void Plan(SimCore sim)
        {
            float lost = 4f * (SimCore.AmberSec + SimCore.AllRedSec);
            float share = sim.opt.freeLeft ? 0.75f : 1f;   // free-left traffic does not need green
            var y = new float[4]; float Y = 0f;
            for (int a = 0; a < 4; a++) { y[a] = sim.rate[a] / 3600f * sim.opt.MeanPcu() * share / Controllers.IrcSatFlowPcuPerSec; Y += y[a]; }
            Y = Mathf.Min(Y, 0.9f);
            cycle = Mathf.Clamp((1.5f * lost + 5f) / (1f - Y), 40f, 150f);
            float sumY = Mathf.Max(1e-3f, y[0] + y[1] + y[2] + y[3]);
            for (int a = 0; a < 4; a++) g[a] = Mathf.Max(7f, (cycle - lost) * y[a] / sumY);
        }

        /// <summary>planned green of an approach (for the countdown display)</summary>
        public float GreenOf(int a) => g[a];

        public void Next(SimCore sim, out int approach, out float green, out string reason)
        {
            Plan(sim);   // re-planned each phase so slider changes are picked up (a real plan is set per time of day)
            approach = sim.t < SimCore.DT * 2 ? 0 : (sim.green + 1) % 4;
            green = g[approach];
            reason = $"Webster plan: cycle {cycle:F0} s, {Geo.Names[approach]} {green:F0} s";
        }

        public bool Hold(SimCore sim, out string reason)
        {
            reason = "planned green over";
            return sim.phaseT < sim.greenDur;
        }
    }

    /// <summary>
    /// Version 1 (modules/smart_signal_controller.py logic): PCU-weighted queue pressure, green sized to clear the
    /// queue, ambulance preempts as soon as it is seen; gap-out and a 100 s fairness limit.
    /// </summary>
    public class SmartController : ISignalController
    {
        public float minGreen = 7f, maxGreen = 50f, maxRedWait = 100f;
        public string Name => "Adaptive v1";

        public void Next(SimCore sim, out int approach, out float g, out string reason)
        {
            var st = sim.app;
            for (int a = 0; a < 4; a++)
                if (st[a].ambulance)
                {
                    approach = a; g = Mathf.Max(12f, GreenFor(st[a].pcu));
                    reason = $"AMBULANCE on {Geo.Names[a]} - priority green";
                    return;
                }

            int best = -1; float bestScore = 0f;
            for (int a = 0; a < 4; a++)
                if (st[a].pcu > 0 && st[a].maxWait > maxRedWait && (best < 0 || st[a].maxWait > st[best].maxWait)) best = a;
            if (best >= 0)
            {
                approach = best; g = GreenFor(st[best].pcu);
                reason = $"{Geo.Names[best]} waited {st[best].maxWait:F0} s (> {maxRedWait:F0} s) - served for fairness";
                return;
            }
            for (int a = 0; a < 4; a++)
            {
                if (a == sim.green && sim.t > 1f && OthersWaiting(st, a)) continue;
                float score = (st[a].pcu + 0.5f * st[a].near) * (1f + st[a].maxWait / 30f);
                if (score > bestScore) { bestScore = score; best = a; }
            }
            if (best < 0)
            {
                approach = (sim.green + 1) % 4; g = minGreen;
                reason = "no demand - short green, round robin";
                return;
            }
            approach = best; g = GreenFor(st[best].pcu + 0.5f * st[best].near);
            reason = $"{Geo.Names[best]}: {st[best].queued} veh / {st[best].pcu:F1} PCU, longest wait {st[best].maxWait:F0} s -> {g:F0} s";
        }

        static bool OthersWaiting(ApproachState[] st, int except)
        {
            for (int a = 0; a < 4; a++) if (a != except && st[a].queued > 0) return true;
            return false;
        }

        float GreenFor(float pcu) => Mathf.Clamp(3f + pcu / Controllers.IrcSatFlowPcuPerSec, minGreen, maxGreen);

        public bool Hold(SimCore sim, out string reason)
        {
            var st = sim.app; int a = sim.green; float el = sim.phaseT;
            reason = "";
            if (st[a].ambulance && el < maxGreen + 20f) return true;
            for (int o = 0; o < 4; o++)
                if (o != a && st[o].ambulance && el >= minGreen) { reason = $"cut short: ambulance on {Geo.Names[o]}"; return false; }
            if (el < minGreen) return true;
            if (st[a].queued == 0 && st[a].near == 0) { reason = "gap-out: approach empty"; return false; }
            if (el < sim.greenDur) return true;
            if (el < maxGreen && st[a].queued > 0) return true;
            reason = el >= maxGreen ? "maximum green reached" : "planned green over";
            return false;
        }
    }

    /// <summary>
    /// PRAVAH - Person-weighted, Red-patience-Aware, Vehicle-mix Adaptive Headway control.
    ///  1. Values people, not vehicles: each waiting vehicle counts its occupants (a bus ~40, a bike ~1.4).
    ///  2. Patience-aware cost: a person who has waited w seconds costs U(w) = 1 + (w / tau)^2 per second, so long
    ///     reds - the cause of red-light jumping - get expensive fast instead of growing linearly.
    ///  3. Self-calibrating discharge: learns each approach's real saturation flow (PCU/s) from the camera during
    ///     queued green, starting from IRC:SP:41 (525 x width), and its arrival rate.
    ///  4. Spillback guard: discounts approaches whose exit arm is full and cuts a green that would feed a blocked box.
    ///  5. Queue-flush ambulance timing: switches to the ambulance's approach when its ETA equals the time to clear
    ///     the queue in front of it, instead of stopping the junction the moment it is seen.
    ///  Decision rule: keep the current green while it relieves more person-urgency per second than the best other
    ///  approach would per second of its service (incl. the 5 s switching loss); otherwise switch to that approach.
    /// </summary>
    public class PravahController : ISignalController
    {
        public string Name => "PRAVAH";
        public float tau = 75f, minGreen = 6f, maxGreen = 50f, jumpRisk = 90f, lostPerSwitch = SimCore.AmberSec + SimCore.AllRedSec;

        public readonly float[] satFlow = { Controllers.IrcSatFlowPcuPerSec, Controllers.IrcSatFlowPcuPerSec, Controllers.IrcSatFlowPcuPerSec, Controllers.IrcSatFlowPcuPerSec };
        public readonly float[] arrival = new float[4];   // learned PCU/s entering the camera zone
        readonly float[] lastArr = new float[4];
        float lastLearnT = -1f, winCrossed, winQueue;
        int winApp = -1;

        struct Q { public int n; public float pcu, persons, urg, occ; }

        float Urg(float w) => 1f + (w / tau) * (w / tau);

        /// <summary>waiting vehicles of approach a; with moving = true also the ones already rolling towards the line
        /// (on the green approach a discharging queue is moving, but it is still the queue being served)</summary>
        Q Measure(SimCore sim, int a, bool moving = false)
        {
            var q = new Q();
            float occW = 0f;
            foreach (var v in sim.vehicles)
            {
                if (v.approach != a || !sim.Visible(v)) continue;
                if (moving ? v.DistToStop > 60f : v.v >= 2f) continue;
                q.n++; q.pcu += v.spec.pcu; q.persons += v.spec.persons; q.urg += v.spec.persons * Urg(v.wait);
                occW += sim.exitOcc[v.exitApproach] * v.spec.pcu;
            }
            q.occ = q.pcu > 0 ? occW / q.pcu : sim.exitOcc[Geo.ExitOf(a, Turn.Straight)];
            return q;
        }

        static float Spill(float occ) => Mathf.Clamp01(1f - (occ - 0.5f) / 0.4f);   // 1 = exit free, 0 = exit full

        float Clear(int a, float pcu) => 2f + pcu / satFlow[a];

        /// <summary>person-urgency relieved per second of cycle time if approach a is served next</summary>
        float Gain(SimCore sim, int a, out Q q)
        {
            q = Measure(sim, a);
            if (q.n == 0) return sim.app[a].near > 0 ? 0.01f * sim.app[a].near : 0f;
            return q.urg * Spill(q.occ) / (Clear(a, q.pcu) + lostPerSwitch);
        }

        void Learn(SimCore sim)
        {
            if (lastLearnT < 0f) { lastLearnT = sim.t; for (int a = 0; a < 4; a++) lastArr[a] = sim.app[a].arrivedPcuTotal; return; }
            float dt = sim.t - lastLearnT;
            if (dt < 1f) return;
            for (int a = 0; a < 4; a++)
            {
                float rate = (sim.app[a].arrivedPcuTotal - lastArr[a]) / dt;
                arrival[a] = Mathf.Lerp(arrival[a], rate, 0.02f * dt);
                lastArr[a] = sim.app[a].arrivedPcuTotal;
            }
            // discharge: 1 s windows of green after the start-up loss, with a standing queue at the window start
            int g = sim.green;
            if (winApp == g && winQueue >= 3f && sim.phase == SimCore.Phase.Green && dt < 1.5f)
            {
                float rate = (sim.app[g].crossedPcuTotal - winCrossed) / dt;
                satFlow[g] = Mathf.Clamp(Mathf.Lerp(satFlow[g], rate, 0.1f), 0.3f, 3f);
            }
            bool ok = sim.phase == SimCore.Phase.Green && sim.phaseT > 3f;
            winApp = ok ? g : -1; winCrossed = sim.app[g].crossedPcuTotal; winQueue = ok ? sim.app[g].pcu : 0f;
            lastLearnT = sim.t;
        }

        int AmbulanceApproach(SimCore sim)
        {
            int best = -1;
            for (int a = 0; a < 4; a++)
                if (sim.app[a].ambulance && (best < 0 || sim.app[a].ambDist < sim.app[best].ambDist)) best = a;
            return best;
        }

        public void Next(SimCore sim, out int approach, out float green, out string reason)
        {
            int amb = AmbulanceApproach(sim);
            if (amb >= 0)
            {
                var qa = Measure(sim, amb);
                approach = amb; green = Mathf.Clamp(Clear(amb, qa.pcu) + 6f, 10f, maxGreen);
                reason = $"PRAVAH: ambulance on {Geo.Names[amb]} {sim.app[amb].ambDist:F0} m out - flushing {qa.n} vehicles ahead of it";
                return;
            }
            // red-patience guard: anyone past ~90 s of red is about to jump it - serve the longest wait first,
            // so person weighting (a full bus) can never starve a few cars or bikes
            int late = -1;
            for (int a = 0; a < 4; a++)
                if (a != sim.green && sim.app[a].maxWait > jumpRisk && (late < 0 || sim.app[a].maxWait > sim.app[late].maxWait)) late = a;
            int best = -1; float bestGain = 0f; Q bq = default;
            for (int a = 0; a < 4; a++)
            {
                if (late >= 0 && a != late) continue;
                float gn = Gain(sim, a, out var q);
                if (a == late) gn = Mathf.Max(gn, 1e-3f);
                if (a == sim.green && sim.t > 1f) gn *= 0.5f;   // just served: needs clearly more need to repeat
                if (gn > bestGain) { bestGain = gn; best = a; bq = q; }
            }
            if (best < 0)
            {
                approach = (sim.green + 1) % 4; green = minGreen;
                reason = "PRAVAH: no one waiting - short green, round robin";
                return;
            }
            float c = Clear(best, bq.pcu);
            approach = best;
            green = Mathf.Clamp(c + arrival[best] * c / satFlow[best], minGreen, maxGreen);
            reason = $"PRAVAH: {Geo.Names[best]} - {bq.n} veh, ~{bq.persons:F0} people, urgency {bq.urg:F0}; clears in ~{c:F0} s " +
                     $"(learned {satFlow[best] * 3600f:F0} PCU/h), exit {bq.occ * 100f:F0}% full";
        }

        public bool Hold(SimCore sim, out string reason)
        {
            Learn(sim);
            int a = sim.green; float el = sim.phaseT;
            reason = "";

            int amb = AmbulanceApproach(sim);
            if (amb == a && el < maxGreen + 30f) return true;
            if (amb >= 0 && el >= minGreen)
            {
                float eta = sim.app[amb].ambDist / Mathf.Max(sim.app[amb].ambSpeed, 6f);
                float need = Clear(amb, Measure(sim, amb).pcu) + lostPerSwitch;
                if (eta <= need + 2f) { reason = $"queue-flush: ambulance on {Geo.Names[amb]} arrives in {eta:F0} s, its queue needs {need:F0} s"; return false; }
            }
            if (el < minGreen) return true;

            var q = Measure(sim, a, moving: true);
            if (Spill(q.occ) < 0.15f && sim.boxBlockers != 0) { reason = $"spillback: {Geo.Names[a]}'s exit is full - stop feeding the box"; return false; }
            if (q.n == 0) { reason = "gap-out: approach empty"; return false; }
            if (el >= maxGreen) { reason = "maximum green reached"; return false; }
            if (el < sim.greenDur) return true;   // finish clearing the queue this green was planned for

            // red-patience guard: past ~90 s of red, riders start jumping - hand over before that
            for (int o = 0; o < 4; o++)
                if (o != a && sim.app[o].maxWait > jumpRisk) { reason = $"patience: {Geo.Names[o]} has waited {sim.app[o].maxWait:F0} s"; return false; }

            float relief = satFlow[a] * (q.urg / Mathf.Max(q.pcu, 0.1f)) * Spill(q.occ);
            float other = 0f; int ob = -1;
            for (int o = 0; o < 4; o++)
            {
                if (o == a) continue;
                float gn = Gain(sim, o, out _);
                if (gn > other) { other = gn; ob = o; }
            }
            if (relief >= other) return true;
            reason = $"switch: {Geo.Names[a]} relieves {relief:F1}/s < {other:F1}/s waiting on {Geo.Names[ob]}";
            return false;
        }
    }

    public struct Scenario
    {
        public string name;
        public float[] rates;
        public float ambPerHour, cattlePerHour;
        public bool busStops;

        public Scenario(string name, float n, float e, float s, float w, float amb, bool busStops = true, float cattle = 0f)
        {
            this.name = name; rates = new[] { n, e, s, w }; ambPerHour = amb; this.busStops = busStops; cattlePerHour = cattle; mix = null;
        }

        public float[] mix;   // optional measured vehicle mix (Specs.Mix order)

        public SimOptions Options() => new SimOptions { busStops = busStops, cattlePerHour = cattlePerHour, mix = mix };

        public static readonly Scenario[] All =
        {
            new Scenario("City balanced", 600, 600, 600, 600, 4),
            new Scenario("Office rush N-S", 1000, 300, 900, 250, 4),
            new Scenario("Market road", 700, 450, 650, 450, 4, true, 6),
            new Scenario("Peak near capacity", 850, 750, 850, 750, 4),
            new Scenario("Night", 200, 150, 200, 120, 2, false),
            new Scenario("Ambulance corridor", 600, 500, 600, 500, 20),
        };
    }

    public struct RunResult
    {
        public string controller, error;
        public float avgWait, personWait, maxWait, ambWait, throughputPerHour, backlogWait, violationsPerHour, boxBlockedPct;
        public int served, ambServed, backlog;
    }

    public static class Compare
    {
        /// <summary>same seed = same arrivals for every controller; warm-up excluded from the stats</summary>
        public static RunResult Run(ISignalController ctl, Scenario sc, int seed, float minutes, float warmupMin = 2f,
                                    System.Action<SimOptions> tweak = null, float demand = 1f)
        {
            var o = sc.Options();
            tweak?.Invoke(o);
            var sim = new SimCore(seed, ctl, o);
            for (int a = 0; a < 4; a++) sim.rate[a] = sc.rates[a] * demand;
            sim.ambulancePerHour = sc.ambPerHour;
            int warm = Mathf.RoundToInt(warmupMin * 60f / SimCore.DT), n = Mathf.RoundToInt(minutes * 60f / SimCore.DT);
            for (int i = 0; i < warm; i++) sim.Step();
            sim.ResetStats();
            for (int i = 0; i < n; i++) sim.Step();
            sim.Backlog(out int bl, out float blw);
            (ctl as System.IDisposable)?.Dispose();
            var st = sim.stats;
            return new RunResult
            {
                controller = ctl.Name, error = (ctl as PythonBridgeController)?.error, avgWait = st.AvgWait, personWait = st.PersonWait, maxWait = st.maxWait, ambWait = st.AmbAvgWait,
                served = st.served, ambServed = st.ambServed, throughputPerHour = st.served / (minutes / 60f),
                violationsPerHour = st.violations / (minutes / 60f), boxBlockedPct = st.boxBlocked / (minutes * 60f) * 100f,
                backlog = bl, backlogWait = blw,
            };
        }
    }
}
