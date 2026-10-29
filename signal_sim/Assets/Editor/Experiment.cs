using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// Headless comparison of every controller over every Indian scenario and several seeds.
    /// Menu: Signal Sim > Run comparison.   Batch: Unity -batchmode -projectPath signal_sim -executeMethod SignalSim.Experiment.Run
    /// Writes Results/compare.md and Results/compare.csv next to Assets/.
    /// </summary>
    public static class Experiment
    {
        const float Minutes = 30f;
        static readonly int[] Seeds = { 1, 2, 3 };

        [MenuItem("Signal Sim/Run comparison")]
        public static void Run()
        {
            string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "Results"));
            Directory.CreateDirectory(dir);
            // -only 2,3 runs just those controllers (quick tuning); the report then covers only them
            var args = System.Environment.GetCommandLineArgs();
            int oi = System.Array.IndexOf(args, "-only");
            var ks = oi >= 0 && oi + 1 < args.Length ? System.Array.ConvertAll(args[oi + 1].Split(','), int.Parse)
                                                     : System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Range(0, Controllers.Bridge));   // the Python bridge only with -only ...,4
            var md = new StringBuilder();
            var csv = new StringBuilder("scenario,seed,controller,avg_wait_s,person_wait_s,max_wait_s,amb_wait_s,amb_served,served,throughput_veh_h,red_jumps_h,box_blocked_pct,queued_at_end\n");
            md.AppendLine($"# Signal controllers on an Indian junction - {Minutes:F0} simulated minutes per run, seeds {string.Join(", ", Seeds)}\n");
            md.AppendLine("Averages over seeds; every controller gets the same arrivals per seed. Wait = time stopped before the stop line; " +
                          "person wait weights each vehicle by its occupants. Ambulance wait averages only runs that had an ambulance.\n");
            md.AppendLine("| Scenario | Controller | Avg wait (s) | Person wait (s) | Worst wait (s) | Ambulance wait (s) | Red-light jumps /h | Box blocked (%) | Throughput (veh/h) | Queued at end |");
            md.AppendLine("|---|---|---|---|---|---|---|---|---|---|");
            foreach (var sc in Scenario.All)
            {
                foreach (int k in ks)
                {
                    var sum = new float[8]; int ambRuns = 0; string lastName = "";
                    foreach (int seed in Seeds)
                    {
                        var r = Compare.Run(Controllers.Make(k), sc, seed, Minutes);
                        if (r.error != null) { Debug.LogError($"[compare] {r.error}"); if (Application.isBatchMode) EditorApplication.Exit(2); return; }
                        csv.AppendLine($"{sc.name},{seed},{r.controller},{r.avgWait:F2},{r.personWait:F2},{r.maxWait:F1},{r.ambWait:F2},{r.ambServed},{r.served},{r.throughputPerHour:F0},{r.violationsPerHour:F1},{r.boxBlockedPct:F1},{r.backlog}");
                        lastName = r.controller; sum[0] += r.avgWait; sum[1] += r.personWait; sum[2] = Mathf.Max(sum[2], r.maxWait);
                        if (r.ambServed > 0) { sum[3] += r.ambWait; ambRuns++; }
                        sum[4] += r.violationsPerHour; sum[5] += r.boxBlockedPct; sum[6] += r.throughputPerHour; sum[7] += r.backlog;
                        Debug.Log($"[compare] {sc.name} seed {seed} {r.controller}: wait {r.avgWait:F1}s person {r.personWait:F1}s amb {r.ambWait:F1}s jumps {r.violationsPerHour:F0}/h box {r.boxBlockedPct:F0}% thr {r.throughputPerHour:F0} backlog {r.backlog}");
                    }
                    float n = Seeds.Length;
                    string amb = ambRuns > 0 ? (sum[3] / ambRuns).ToString("F1") : "-";
                    md.AppendLine($"| {sc.name} | {(k == Controllers.Bridge ? lastName : Controllers.Names[k])} | {sum[0] / n:F1} | {sum[1] / n:F1} | {sum[2]:F0} | {amb} | {sum[4] / n:F0} | {sum[5] / n:F0} | {sum[6] / n:F0} | {sum[7] / n:F0} |");
                }
            }
            File.WriteAllText(System.IO.Path.Combine(dir, "compare.md"), md.ToString());
            File.WriteAllText(System.IO.Path.Combine(dir, "compare.csv"), csv.ToString());
            Debug.Log($"[compare] DONE -> {dir}");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
    }
}
