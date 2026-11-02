using System;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace SignalSim
{
    /// <summary>
    /// The signal is decided by a Python process (signal_bridge/server.py): either the project's own
    /// modules/smart_signal_controller.py or the Python port of PRAVAH. Each step Unity sends what the junction cameras
    /// see; Python answers. If Python is not running the junction falls back to a fixed 30 s cycle and says so.
    /// </summary>
    public class PythonBridgeController : ISignalController, IDisposable
    {
        public static int Port = 5055;
        string name = "Python bridge";
        public string Name => name;
        public string error;

        TcpClient client;
        StreamReader reader;
        StreamWriter writer;
        readonly FixedTimeController fallback = new FixedTimeController();
        readonly StringBuilder sb = new StringBuilder(8192);
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        [Serializable] class Reply { public string name; public int approach; public float green; public string reason; public bool hold; }

        public PythonBridgeController()
        {
            try
            {
                client = new TcpClient { NoDelay = true, ReceiveTimeout = 10000, SendTimeout = 10000 };
                client.Connect("127.0.0.1", Port);
                var s = client.GetStream();
                reader = new StreamReader(s, new UTF8Encoding(false));
                writer = new StreamWriter(s, new UTF8Encoding(false)) { AutoFlush = false, NewLine = "\n" };
                var hello = JsonUtility.FromJson<Reply>(reader.ReadLine());
                name = hello.name;
            }
            catch (Exception e) { Fail(e); }
        }

        void Fail(Exception e)
        {
            error = $"Python bridge offline ({e.GetType().Name}) - start: python3 signal_bridge/server.py";
            name = "Python bridge (OFFLINE)";
            Dispose();
        }

        Reply Ask(SimCore sim, string op)
        {
            sb.Clear();
            sb.Append("{\"op\":\"").Append(op).Append("\",\"t\":").Append(F(sim.t)).Append(",\"green\":").Append(sim.green)
              .Append(",\"phase\":\"").Append(sim.phase).Append("\",\"phase_t\":").Append(F(sim.phaseT)).Append(",\"green_dur\":").Append(F(sim.greenDur))
              .Append(",\"box\":").Append(sim.boxBlockers).Append(",\"app\":[");
            for (int a = 0; a < 4; a++)
            {
                var s = sim.app[a];
                if (a > 0) sb.Append(',');
                sb.Append("{\"pcu\":").Append(F(s.pcu)).Append(",\"persons\":").Append(F(s.persons)).Append(",\"max_wait\":").Append(F(s.maxWait))
                  .Append(",\"near\":").Append(s.near).Append(",\"amb\":").Append(s.ambulance ? "true" : "false")
                  .Append(",\"amb_dist\":").Append(F(s.ambulance ? s.ambDist : 0f)).Append(",\"amb_speed\":").Append(F(s.ambSpeed))
                  .Append(",\"crossed_pcu\":").Append(F(s.crossedPcuTotal)).Append(",\"arrived_pcu\":").Append(F(s.arrivedPcuTotal)).Append('}');
            }
            sb.Append("],\"exit_occ\":[");
            for (int b = 0; b < 4; b++) { if (b > 0) sb.Append(','); sb.Append(F(sim.exitOcc[b])); }
            sb.Append("],\"veh\":[");
            bool first = true;
            foreach (var v in sim.vehicles)
            {
                if (!sim.Visible(v)) continue;
                if (!first) sb.Append(','); first = false;
                sb.Append("{\"id\":").Append(v.id).Append(",\"a\":").Append(v.approach).Append(",\"type\":\"").Append(v.type)
                  .Append("\",\"pcu\":").Append(F(v.spec.pcu)).Append(",\"persons\":").Append(F(v.spec.persons)).Append(",\"wait\":").Append(F(v.wait))
                  .Append(",\"v\":").Append(F(v.v)).Append(",\"dist\":").Append(F(v.DistToStop)).Append(",\"exit\":").Append(v.exitApproach)
                  .Append(",\"jumper\":").Append(v.jumper ? 1 : 0).Append('}');
            }
            sb.Append("]}");
            writer.WriteLine(sb.ToString());
            writer.Flush();
            return JsonUtility.FromJson<Reply>(reader.ReadLine());
        }

        static string F(float x) => float.IsInfinity(x) || float.IsNaN(x) ? "0" : x.ToString("R", Inv);

        /// <summary>ask Python without acting on the answer (shadow testing): (approach, green) for next, hold flag for hold</summary>
        public string lastReason;
        public bool Shadow(SimCore sim, string op, out int approach, out float green, out bool hold)
        {
            approach = -1; green = 0; hold = false;
            if (client == null) return false;
            var r = Ask(sim, op);
            approach = r.approach; green = r.green; hold = r.hold; lastReason = r.reason;
            return true;
        }

        public void Next(SimCore sim, out int approach, out float greenSec, out string reason)
        {
            if (client != null)
                try
                {
                    var r = Ask(sim, "next");
                    approach = Mathf.Clamp(r.approach, 0, 3); greenSec = r.green; reason = r.reason;
                    return;
                }
                catch (Exception e) { Fail(e); }
            fallback.Next(sim, out approach, out greenSec, out _);
            reason = error;
        }

        public bool Hold(SimCore sim, out string reason)
        {
            if (client != null)
                try
                {
                    var r = Ask(sim, "hold");
                    reason = r.reason;
                    return r.hold;
                }
                catch (Exception e) { Fail(e); }
            bool h = fallback.Hold(sim, out _);
            reason = error;
            return h;
        }

        public void Dispose()
        {
            try { reader?.Dispose(); writer?.Dispose(); client?.Close(); } catch { }
            reader = null; writer = null; client = null;
        }
    }
}
