using UnityEngine;

namespace SignalSim
{
    public enum CamMode { Orbit, Fly, TopDown, Cctv, ControlRoom, Follow, Ambulance, Drone }

    /// <summary>All the ways to look at the junction. Keys 1-8 pick a mode; mouse/keyboard controls depend on the mode.</summary>
    public class CameraRig
    {
        public static readonly string[] ModeNames = { "Orbit", "Free fly", "Top-down", "CCTV", "Control room", "Follow", "Ambulance", "Drone" };
        public CamMode mode = CamMode.Orbit;
        public Camera main, mini;
        public readonly Camera[] cctv = new Camera[4];
        public int cctvIndex;
        public bool driverView, showMinimap = true;
        public Vehicle follow;

        Vector3 target = Vector3.zero;
        float yaw = 25f, pitch = 50f, dist = 110f;
        Vector3 flyPos = new Vector3(0, 25, -70);
        float flyYaw, flyPitch = 20f, followYaw, followDist = 1f, droneT;

        public void Init(Transform parent, WorldBuilder world)
        {
            main = Camera.main;
            if (main == null) { main = new GameObject("Main Camera").AddComponent<Camera>(); main.tag = "MainCamera"; }
            main.clearFlags = CameraClearFlags.Skybox;
            main.farClipPlane = 1500f; main.nearClipPlane = 0.1f; main.fieldOfView = 55f;
            if (main.GetComponent<AudioListener>() == null) main.gameObject.AddComponent<AudioListener>();
            for (int a = 0; a < 4; a++)
            {
                var c = new GameObject($"CCTV {Geo.Names[a]}").AddComponent<Camera>();
                c.transform.SetParent(parent, false);
                c.transform.SetPositionAndRotation(world.cctvPos[a], Quaternion.LookRotation(world.cctvLook[a] - world.cctvPos[a]));
                c.fieldOfView = 48f; c.farClipPlane = 400f; c.enabled = false; c.depth = 1;
                cctv[a] = c;
            }
            mini = new GameObject("Minimap").AddComponent<Camera>();
            mini.transform.SetParent(parent, false);
            mini.transform.SetPositionAndRotation(new Vector3(0, 200, 0), Quaternion.Euler(90, 0, 0));
            mini.orthographic = true; mini.orthographicSize = 75f; mini.depth = 5; mini.farClipPlane = 400f;
            mini.clearFlags = CameraClearFlags.SolidColor; mini.backgroundColor = new Color(0.2f, 0.2f, 0.2f);
            mini.rect = new Rect(0.79f, 0.66f, 0.2f, 0.32f);
            Post.SetupCamera(main, true);
            foreach (var c in cctv) Post.SetupCamera(c, true);
            Post.SetupCamera(mini, false);
        }

        public void SetMode(CamMode m)
        {
            if (m == CamMode.Fly && mode != CamMode.Fly)
            {
                flyPos = main.transform.position;
                var e = main.transform.eulerAngles; flyYaw = e.y; flyPitch = e.x > 180 ? e.x - 360 : e.x;
            }
            if (m == CamMode.Cctv && mode == CamMode.Cctv) cctvIndex = (cctvIndex + 1) % 4;   // pressing again cycles cameras
            mode = m;
        }

        public void Update(SimCore sim, float dt, bool mouseOverUi)
        {
            bool rmb = Input.GetMouseButton(1) && !mouseOverUi, mmb = Input.GetMouseButton(2) && !mouseOverUi;
            float mx = Input.GetAxis("Mouse X"), my = Input.GetAxis("Mouse Y");
            float scroll = mouseOverUi ? 0f : Input.mouseScrollDelta.y;
            float k = 1f - Mathf.Exp(-dt * 6f);

            if ((mode == CamMode.Follow || mode == CamMode.Ambulance) && (follow == null || !sim.vehicles.Contains(follow)))
            {
                follow = null;
                if (mode == CamMode.Ambulance)
                    foreach (var v in sim.vehicles) if (v.type == VType.Ambulance) { follow = v; break; }
                if (follow == null && mode == CamMode.Follow) PickNext(sim);
            }

            bool multi = mode == CamMode.ControlRoom;
            main.enabled = !multi && mode != CamMode.Cctv;
            for (int a = 0; a < 4; a++)
            {
                cctv[a].enabled = multi || (mode == CamMode.Cctv && a == cctvIndex);
                cctv[a].rect = multi ? new Rect((a % 2) * 0.5f, a < 2 ? 0.5f : 0f, 0.5f, 0.5f) : new Rect(0, 0, 1, 1);
            }
            mini.enabled = showMinimap && !multi;

            bool chasing = (mode == CamMode.Follow || mode == CamMode.Ambulance) && follow != null;
            Post.Focus(chasing ? Vector3.Distance(main.transform.position, follow.pos) : 0f);
            switch (mode)
            {
                case CamMode.Orbit:
                    if (rmb) { yaw += mx * 3f; pitch = Mathf.Clamp(pitch - my * 3f, 5f, 89f); }
                    var flat = Quaternion.Euler(0, yaw, 0);
                    Vector3 pan = flat * new Vector3(Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical")) * dist * 0.8f * dt;
                    if (mmb) pan += flat * new Vector3(-mx, 0, -my) * dist * 0.02f;
                    target += pan;
                    target = new Vector3(Mathf.Clamp(target.x, -150, 150), 0, Mathf.Clamp(target.z, -150, 150));
                    dist = Mathf.Clamp(dist * (1f - scroll * 0.1f), 8f, 350f);
                    Place(target + Quaternion.Euler(pitch, yaw, 0) * new Vector3(0, 0, -dist), Quaternion.Euler(pitch, yaw, 0), k);
                    break;

                case CamMode.Fly:
                    if (rmb) { flyYaw += mx * 3f; flyPitch = Mathf.Clamp(flyPitch - my * 3f, -89f, 89f); }
                    var r = Quaternion.Euler(flyPitch, flyYaw, 0);
                    float sp = (Input.GetKey(KeyCode.LeftShift) ? 60f : 15f) * dt;
                    flyPos += r * new Vector3(Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical")) * sp;
                    if (Input.GetKey(KeyCode.E)) flyPos += Vector3.up * sp;
                    if (Input.GetKey(KeyCode.Q)) flyPos -= Vector3.up * sp;
                    flyPos += r * Vector3.forward * scroll * 3f;
                    flyPos.y = Mathf.Max(flyPos.y, 0.5f);
                    main.transform.SetPositionAndRotation(flyPos, r);
                    break;

                case CamMode.TopDown:
                    dist = Mathf.Clamp(dist * (1f - scroll * 0.1f), 30f, 350f);
                    Place(new Vector3(target.x, dist * 1.3f, target.z), Quaternion.Euler(90, 0, 0), k);
                    break;

                case CamMode.Follow:
                case CamMode.Ambulance:
                    if (follow == null) { Place(new Vector3(0, 60, -80), Quaternion.Euler(35, 0, 0), k); break; }
                    if (rmb) followYaw += mx * 3f; else followYaw = Mathf.Lerp(followYaw, 0f, dt * 1.5f);
                    followDist = Mathf.Clamp(followDist * (1f - scroll * 0.1f), 0.4f, 4f);
                    var fd = Quaternion.Euler(0, followYaw, 0) * follow.dir;
                    if (driverView && mode == CamMode.Follow)
                    {
                        // Indian vehicles are right-hand drive: the driver sits on the right of the vehicle
                        Vector3 eye = follow.pos + follow.dir * follow.spec.len * 0.08f - Geo.Left(follow.dir) * follow.spec.wid * 0.22f
                                      + Vector3.up * Mathf.Max(1.25f, follow.spec.height * 0.85f);
                        if (follow.type == VType.Bike || follow.type == VType.Bicycle) eye = follow.pos + Vector3.up * 1.9f;
                        main.transform.SetPositionAndRotation(eye, Quaternion.LookRotation(fd + Vector3.down * 0.05f));
                    }
                    else
                    {
                        float back = (follow.spec.len * 1.2f + 7f) * followDist;
                        Vector3 cp = follow.pos - fd * back + Vector3.up * (2.5f + back * 0.3f);
                        Place(cp, Quaternion.LookRotation(follow.pos + follow.dir * 8f + Vector3.up - cp), k * 1.5f);
                    }
                    break;

                case CamMode.Drone:
                    droneT += dt;
                    float dy = droneT * 8f;
                    Vector3 dp = Quaternion.Euler(0, dy, 0) * new Vector3(0, 0, -95f) + Vector3.up * (40f + Mathf.Sin(droneT * 0.2f) * 15f);
                    Place(dp, Quaternion.LookRotation(-dp + Vector3.up * 10f), k);
                    break;
            }
        }

        void Place(Vector3 pos, Quaternion rot, float k)
        {
            var t = main.transform;
            t.SetPositionAndRotation(Vector3.Lerp(t.position, pos, k), Quaternion.Slerp(t.rotation, rot, k));
        }

        /// <summary>next vehicle to follow, preferring ones that are on the approach and moving</summary>
        public void PickNext(SimCore sim)
        {
            if (sim.vehicles.Count == 0) return;
            int start = follow != null ? sim.vehicles.IndexOf(follow) : -1;
            for (int i = 1; i <= sim.vehicles.Count; i++)
            {
                var v = sim.vehicles[(start + i + sim.vehicles.Count) % sim.vehicles.Count];
                if (!v.Obstacle && !v.Crossed && v.s > 15f) { follow = v; return; }
            }
            follow = sim.vehicles[0];
        }

        /// <summary>vehicle nearest the mouse click on screen</summary>
        public Vehicle Pick(SimCore sim, Vector2 mouse)
        {
            Camera cam = mode == CamMode.Cctv ? cctv[cctvIndex] : main;
            Vehicle best = null; float bd = 45f;
            foreach (var v in sim.vehicles)
            {
                if (v.Obstacle) continue;
                var sp = cam.WorldToScreenPoint(v.pos + Vector3.up * v.spec.height * 0.5f);
                if (sp.z <= 0) continue;
                float d = Vector2.Distance(mouse, sp);
                if (d < bd) { bd = d; best = v; }
            }
            return best;
        }
    }
}
