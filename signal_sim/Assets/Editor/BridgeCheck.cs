using UnityEditor;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// Shadow test of the Python PRAVAH port: C# PRAVAH runs the signal, Python gets the identical state at every
    /// decision, and every disagreement is counted. Needs signal_bridge/server.py --controller pravah running.
    /// Batch: -executeMethod SignalSim.BridgeCheck.Run
    /// </summary>
    public static class BridgeCheck
    {
        class Shadowed : ISignalController
        {
            public readonly PravahController cs = new PravahController();
            public readonly PythonBridgeController py = new PythonBridgeController();
            public int nextN, nextDiff, holdN, holdDiff; public float greenErr;
            public string Name => "shadow";

            public void Next(SimCore sim, out int a, out float g, out string why)
            {
                py.Shadow(sim, "next", out int pa, out float pg, out _);
                cs.Next(sim, out a, out g, out why);
                nextN++; if (pa != a) nextDiff++; else greenErr = Mathf.Max(greenErr, Mathf.Abs(pg - g));
            }

            public bool Hold(SimCore sim, out string why)
            {
                py.Shadow(sim, "hold", out _, out _, out bool ph);
                bool h = cs.Hold(sim, out why);
                holdN++; if (ph != h) { holdDiff++; if (holdDiff <= 4) Debug.Log($"[bridgecheck] t={sim.t:F1} green={sim.green} phaseT={sim.phaseT:F1} dur={sim.greenDur:F1} C#={h} '{why}' Py={ph} '{py.lastReason}'"); }
                return h;
            }
        }

        public static void Run()
        {
            foreach (var sc in Scenario.All)
            {
                var ctl = new Shadowed();
                if (ctl.py.error != null) { Debug.LogError("[bridgecheck] " + ctl.py.error); break; }
                Compare.Run(ctl, sc, 1, 30f);
                ctl.py.Dispose();
                Debug.Log($"[bridgecheck] {sc.name}: next {ctl.nextDiff}/{ctl.nextN} differ (max green diff {ctl.greenErr:F2} s), hold {ctl.holdDiff}/{ctl.holdN} differ");
            }
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
    }
}
