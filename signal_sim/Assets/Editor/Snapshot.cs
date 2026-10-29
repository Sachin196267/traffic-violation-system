using UnityEditor;
using UnityEngine;

namespace SignalSim
{
    /// <summary>Render the junction from several camera modes to PNGs without entering Play mode.
    /// Batch (with graphics): -executeMethod SignalSim.Snapshot.Run</summary>
    public static class Snapshot
    {
        public static void Run()
        {
            string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "Results"));
            System.IO.Directory.CreateDirectory(dir);
            ShaderUtil.allowAsyncCompilation = false;   // otherwise the first frames render with placeholder shaders
            foreach (var (tod, tag, wet) in new[] { (0f, "day", false), (1f, "night", false), (0f, "rain", true) })
            {
                var go = new GameObject("SignalSim");
                var view = go.AddComponent<SimView>();
                view.PrepareSnapshot(3, 3f, tod, wet);
                if (wet) for (int i = 0; i < 40; i++) foreach (var ps in Object.FindObjectsByType<ParticleSystem>(FindObjectsSortMode.None)) ps.Simulate(0.05f, true, false);
                var shots = tag == "day"
                    ? new[] { (CamMode.Orbit, "overview"), (CamMode.Cctv, "cctv"), (CamMode.ControlRoom, "control_room"), (CamMode.Follow, "chase"), (CamMode.Ambulance, "ambulance"), (CamMode.TopDown, "topdown") }
                    : new[] { (CamMode.Orbit, "overview"), (CamMode.Follow, "chase") };
                foreach (var (mode, name) in shots)
                {
                    var tex = view.Capture(mode, 1600, 900);
                    System.IO.File.WriteAllBytes(System.IO.Path.Combine(dir, $"snap_{tag}_{name}.png"), tex.EncodeToPNG());
                }
                foreach (var o in Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None)) if (o) Object.DestroyImmediate(o);
            }
            Debug.Log("[snapshot] DONE");
            if (Application.isBatchMode) EditorApplication.Exit(0);
        }
    }
}
