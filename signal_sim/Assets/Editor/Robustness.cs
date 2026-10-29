using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// Robustness of the controller comparison: every scenario x 10 seeds, repeated with each simulator assumption
    /// changed (driver patience, bus dwell, camera range, demand, vehicle mix, red-light jumping, free left).
    /// Writes Results/robustness.csv (one row per run); analysis: Tools/robustness_report.py.
    /// Batch: -executeMethod SignalSim.Robustness.Run [-seeds 10] [-minutes 30]
    /// </summary>
    public static class Robustness
    {
        static readonly (string name, Action<SimOptions> tweak, float demand)[] Conditions =
        {
            ("baseline", null, 1f),
            ("impatient drivers (patience x0.5)", o => o.patienceScale = 0.5f, 1f),
            ("patient drivers (patience x1.5)", o => o.patienceScale = 1.5f, 1f),
            ("short bus stops (dwell x0.5)", o => o.dwellScale = 0.5f, 1f),
            ("long bus stops (dwell x2)", o => o.dwellScale = 2f, 1f),
            ("short camera range (50 m)", o => { o.detectRange = 50f; o.ambulanceRange = 50f; }, 1f),
            ("long camera range (150 m)", o => { o.detectRange = 150f; o.ambulanceRange = 150f; }, 1f),
            ("demand -20%", null, 0.8f),
            ("demand +20%", null, 1.2f),
            ("two-wheeler heavy mix (58%)", o => o.mix = new[] { 0.58f, 0.04f, 0.18f, 0.10f, 0.05f, 0.05f }, 1f),
            ("no red-light jumping", o => o.redJumping = false, 1f),
            ("no free left turn", o => o.freeLeft = false, 1f),
        };

        static string Arg(string n, string d) { var a = Environment.GetCommandLineArgs(); int i = Array.IndexOf(a, n); return i >= 0 && i + 1 < a.Length ? a[i + 1] : d; }

        public static void Run()
        {
            int seeds = int.Parse(Arg("-seeds", "10"));
            float minutes = float.Parse(Arg("-minutes", "30"), System.Globalization.CultureInfo.InvariantCulture);
            string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "Results"));
            Directory.CreateDirectory(dir);
            string path = System.IO.Path.Combine(dir, "robustness.csv");
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            using (var w = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                w.WriteLine("condition,scenario,seed,controller,avg_wait,person_wait,max_wait,amb_wait,amb_served,red_jumps_h,box_pct,throughput_h,queued_end");
                int done = 0, total = Conditions.Length * Scenario.All.Length * seeds;
                var t0 = DateTime.Now;
                foreach (var c in Conditions)
                    foreach (var sc in Scenario.All)
                        for (int seed = 1; seed <= seeds; seed++)
                        {
                            for (int k = 0; k < Controllers.Bridge; k++)
                            {
                                if (k == 0 && c.name != "baseline") continue;   // the naive fixed cycle only in the baseline
                                var r = Compare.Run(Controllers.Make(k), sc, seed, minutes, 2f, c.tweak, c.demand);
                                w.WriteLine(string.Join(",", $"\"{c.name}\"", sc.name, seed, Controllers.Names[k],
                                    r.avgWait.ToString("F2", inv), r.personWait.ToString("F2", inv), r.maxWait.ToString("F1", inv), r.ambWait.ToString("F2", inv),
                                    r.ambServed, r.violationsPerHour.ToString("F1", inv), r.boxBlockedPct.ToString("F2", inv), r.throughputPerHour.ToString("F0", inv), r.backlog));
                            }
                            w.Flush();
                            if (++done % 20 == 0) Debug.Log($"[robust] {done}/{total} ({(DateTime.Now - t0).TotalMinutes:F1} min)");
                        }
            }
            Debug.Log("[robust] DONE -> " + path);
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
    }
}
