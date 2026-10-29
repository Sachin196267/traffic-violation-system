using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// One simulation request from the web platform (app/server/signal_sim.py): traffic measured from an analysed video
    /// plus the user's assumptions -> every built-in controller on identical arrivals -> JSON result.
    /// Batch: -executeMethod SignalSim.SimJob.Run -job job.json -out result.json
    /// </summary>
    public static class SimJob
    {
        [Serializable] public class Job { public float[] rates; public float[] mix; public float ambPerHour; public bool busStops; public float cattlePerHour; public float minutes = 20f; public int[] seeds = { 1, 2, 3 }; }
        [Serializable] public class Row { public string controller; public float avg_wait, person_wait, max_wait, amb_wait, red_jumps_per_hour, box_blocked_pct, throughput_per_hour, queued_at_end; public int amb_runs; }
        [Serializable] public class Result { public Row[] rows; public float minutes; public int[] seeds; public string unity; }

        static string Arg(string name)
        {
            var a = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(a, name);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : null;
        }

        public static void Run()
        {
            string jobPath = Arg("-job"), outPath = Arg("-out");
            int code = 0;
            try
            {
                var job = JsonUtility.FromJson<Job>(File.ReadAllText(jobPath));
                if (job.rates == null || job.rates.Length != 4) throw new Exception("job.rates must have 4 values (N, E, S, W)");
                var sc = new Scenario("Measured", job.rates[0], job.rates[1], job.rates[2], job.rates[3], job.ambPerHour, job.busStops, job.cattlePerHour) { mix = job.mix };
                var rows = new Row[Controllers.Bridge];
                for (int k = 0; k < rows.Length; k++)
                {
                    var r = new Row { controller = Controllers.Names[k] };
                    foreach (int seed in job.seeds)
                    {
                        var x = Compare.Run(Controllers.Make(k), sc, seed, job.minutes);
                        float n = job.seeds.Length;
                        r.avg_wait += x.avgWait / n; r.person_wait += x.personWait / n; r.max_wait = Mathf.Max(r.max_wait, x.maxWait);
                        r.red_jumps_per_hour += x.violationsPerHour / n; r.box_blocked_pct += x.boxBlockedPct / n;
                        r.throughput_per_hour += x.throughputPerHour / n; r.queued_at_end += x.backlog / n;
                        if (x.ambServed > 0) { r.amb_wait += x.ambWait; r.amb_runs++; }
                    }
                    if (r.amb_runs > 0) r.amb_wait /= r.amb_runs;
                    rows[k] = r;
                    Debug.Log($"[simjob] {r.controller}: wait {r.avg_wait:F1} s");
                }
                File.WriteAllText(outPath, JsonUtility.ToJson(new Result { rows = rows, minutes = job.minutes, seeds = job.seeds, unity = Application.unityVersion }));
            }
            catch (Exception e) { Debug.LogError("[simjob] " + e); code = 1; }
            if (Application.isBatchMode) EditorApplication.Exit(code);
        }
    }
}
