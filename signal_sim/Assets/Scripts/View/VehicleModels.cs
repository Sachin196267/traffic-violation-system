using System.Collections.Generic;
using UnityEngine;

namespace SignalSim
{
    /// <summary>A built vehicle model and the parts that animate: wheels, brake/head/indicator lights, ambulance beacons.</summary>
    public class VehicleRig
    {
        public GameObject go;
        public readonly List<Transform> wheels = new List<Transform>();
        public float wheelR = 0.3f;
        public readonly List<Renderer> brake = new List<Renderer>(), head = new List<Renderer>(), indL = new List<Renderer>(), indR = new List<Renderer>();
        public Renderer beaconA, beaconB;
        public UnityEngine.Light beaconLight;
        public string plate;
        float spin;
        int brakeState = -1, headState = -1, indState = -1, beaconState = -1;

        static Material TailOff => Mats.Solid(new Color(0.4f, 0.04f, 0.04f), 0.7f);
        static Material TailDim => Mats.Glow(new Color(0.9f, 0.05f, 0.05f), 1.2f);
        static Material TailOn => Mats.Glow(new Color(1f, 0.08f, 0.05f), 4f);
        static Material HeadOff => Mats.Solid(new Color(0.85f, 0.85f, 0.8f), 0.9f);
        static Material HeadOn => Mats.Glow(new Color(1f, 0.95f, 0.8f), 3f);
        static Material IndOff => Mats.Solid(new Color(0.55f, 0.35f, 0.05f), 0.7f);
        static Material IndOn => Mats.Glow(new Color(1f, 0.6f, 0.05f), 4f);

        public void Update(Vehicle v, float dt, float night, float time)
        {
            go.transform.SetPositionAndRotation(v.pos, Quaternion.LookRotation(v.dir));
            if (wheels.Count > 0)
            {
                spin = (spin + v.v * dt / wheelR * Mathf.Rad2Deg) % 360f;
                foreach (var w in wheels) w.localRotation = Quaternion.Euler(spin, 0f, 0f);
            }
            int b = v.acc < -1f || v.v < 0.3f ? 2 : night > 0.3f ? 1 : 0;
            if (b != brakeState) { brakeState = b; var m = b == 2 ? TailOn : b == 1 ? TailDim : TailOff; foreach (var r in brake) r.sharedMaterial = m; }
            int h = night > 0.3f ? 1 : 0;
            if (h != headState) { headState = h; foreach (var r in head) r.sharedMaterial = h == 1 ? HeadOn : HeadOff; }
            bool turning = v.turn != Turn.Straight && (v.Crossed ? v.s < v.sStop + 30f : v.DistToStop < 50f);
            bool blink = turning && (time * 2.5f) % 1f < 0.5f;
            int ind = !blink ? 0 : v.turn == Turn.Left ? 1 : 2;
            if (ind != indState)
            {
                indState = ind;
                foreach (var r in indL) r.sharedMaterial = ind == 1 ? IndOn : IndOff;
                foreach (var r in indR) r.sharedMaterial = ind == 2 ? IndOn : IndOff;
            }
            if (beaconA != null)
            {
                int s = (time * 4f) % 2f < 1f ? 0 : 1;
                if (s != beaconState)
                {
                    beaconState = s;
                    beaconA.sharedMaterial = s == 0 ? Mats.Glow(Color.red, 5f) : Mats.Solid(new Color(0.4f, 0.05f, 0.05f), 0.8f);
                    beaconB.sharedMaterial = s == 1 ? Mats.Glow(new Color(0.1f, 0.3f, 1f), 5f) : Mats.Solid(new Color(0.05f, 0.08f, 0.4f), 0.8f);
                    if (beaconLight) beaconLight.color = s == 0 ? Color.red : new Color(0.2f, 0.4f, 1f);
                }
                if (beaconLight) beaconLight.intensity = Mathf.Lerp(1.5f, 4f, night);
            }
        }
    }

    public static class VehicleModels
    {
        static readonly string[] States = { "MH", "MH", "KA", "DL", "TN", "TS", "GJ", "UP", "KL", "RJ", "WB", "AP", "HR", "MP" };
        static readonly string[] Routes = { "245 STATION", "83 MARKET", "12 BUS DEPOT", "501 AIRPORT", "37 UNIVERSITY", "8 CITY CENTRE" };

        static Material M(Color c, float gloss = 0.15f, float metal = 0f) => Mats.Solid(c, gloss, metal);
        static Material Glass => M(new Color(0.06f, 0.08f, 0.1f), 0.97f, 0.1f);
        static Material Tyre => M(new Color(0.07f, 0.07f, 0.07f), 0.05f);
        static Material Rim => M(new Color(0.62f, 0.62f, 0.64f), 0.6f, 0.6f);
        static Material Black => M(new Color(0.06f, 0.06f, 0.06f), 0.3f);
        static Material Skin(int id) => M(new[] { new Color(0.55f, 0.38f, 0.26f), new Color(0.45f, 0.3f, 0.2f), new Color(0.65f, 0.47f, 0.33f) }[id % 3], 0.2f);
        static Material Cloth(int id) => M(new[] { new Color(0.2f, 0.3f, 0.6f), new Color(0.7f, 0.2f, 0.2f), new Color(0.93f, 0.93f, 0.9f),
            new Color(0.3f, 0.3f, 0.3f), new Color(0.9f, 0.55f, 0.15f), new Color(0.45f, 0.6f, 0.35f) }[id % 6], 0.1f);

        public static string PlateText(int id)
        {
            var r = new System.Random(id * 7919 + 13);
            string st = States[r.Next(States.Length)];
            string series = r.NextDouble() < 0.15 ? ((char)('A' + r.Next(26))).ToString() : $"{(char)('A' + r.Next(26))}{(char)('A' + r.Next(26))}";
            return $"{st} {r.Next(1, 50):00} {series} {r.Next(1, 10000):0000}";
        }

        public static VehicleRig Build(Vehicle v, Transform parent)
        {
            var rig = new VehicleRig { go = new GameObject($"{v.type} {v.id}") };
            rig.go.transform.SetParent(parent, false);
            var t = rig.go.transform;
            switch (v.type)
            {
                case VType.Car: Car(v, t, rig); break;
                case VType.Bike: Bike(v, t, rig, false); break;
                case VType.Bicycle: Bike(v, t, rig, true); break;
                case VType.Auto: Auto(v, t, rig); break;
                case VType.Bus: Bus(v, t, rig); break;
                case VType.Truck: Truck(v, t, rig); break;
                case VType.Ambulance: Ambulance(v, t, rig); break;
                case VType.Cattle: Cattle(v, t); break;
            }
            return rig;
        }

        /// <summary>parked two-wheeler for street dressing</summary>
        public static GameObject ParkedBike(Transform parent, Vector3 pos, float yaw, int id)
        {
            var g = new GameObject("Parked bike").transform;
            g.SetParent(parent, false); g.localPosition = pos; g.localRotation = Quaternion.Euler(0, yaw, 0);
            var paint = M(new[] { Color.black, new Color(0.6f, 0.05f, 0.05f), new Color(0.1f, 0.2f, 0.5f), new Color(0.75f, 0.75f, 0.78f) }[id % 4], 0.6f);
            Shapes.Box(g, new Vector3(0, 0.6f, 0), new Vector3(0.22f, 0.32f, 1.15f), paint);
            Shapes.Box(g, new Vector3(0, 0.85f, 0.25f), new Vector3(0.3f, 0.2f, 0.45f), paint);
            Shapes.Box(g, new Vector3(0, 0.86f, -0.25f), new Vector3(0.28f, 0.1f, 0.6f), Black);
            Wheel(g, 0, 0.68f, 0.3f, 0.1f, null); Wheel(g, 0, -0.68f, 0.3f, 0.1f, null);
            return g.gameObject;
        }

        static Transform Wheel(Transform p, float x, float z, float r, float w, VehicleRig rig)
        {
            var pivot = new GameObject("Wheel").transform;
            pivot.SetParent(p, false);
            pivot.localPosition = new Vector3(x, r, z);
            Shapes.Part(pivot, PrimitiveType.Cylinder, Vector3.zero, new Vector3(2 * r, w * 0.5f, 2 * r), Tyre, new Vector3(0, 0, 90));
            Shapes.Part(pivot, PrimitiveType.Cylinder, new Vector3(Mathf.Sign(x == 0 ? 1 : x) * 0.01f, 0, 0), new Vector3(1.2f * r, w * 0.52f, 1.2f * r), Rim, new Vector3(0, 0, 90), false);
            Shapes.Box(pivot, Vector3.zero, new Vector3(w * 1.05f, 0.08f, 1.5f * r), Rim, default, false);   // spoke, makes spin visible
            if (rig != null) { rig.wheels.Add(pivot); rig.wheelR = r; }
            return pivot;
        }

        static void Lights(Transform t, VehicleRig rig, float wid, float len, float yHead, float yTail, float spread = 0.33f)
        {
            foreach (float s in new[] { -1f, 1f })
            {
                rig.head.Add(Shapes.Box(t, new Vector3(s * wid * spread, yHead, len / 2 + 0.01f), new Vector3(0.3f, 0.13f, 0.05f), HeadOffMat(), default, false).GetComponent<Renderer>());
                rig.brake.Add(Shapes.Box(t, new Vector3(s * wid * 0.36f, yTail, -len / 2 - 0.01f), new Vector3(0.28f, 0.14f, 0.05f), TailOffMat(), default, false).GetComponent<Renderer>());
                var ind = s < 0 ? rig.indL : rig.indR;
                ind.Add(Shapes.Box(t, new Vector3(s * wid * 0.46f, yHead, len / 2 + 0.01f), new Vector3(0.1f, 0.09f, 0.05f), IndOffMat(), default, false).GetComponent<Renderer>());
                ind.Add(Shapes.Box(t, new Vector3(s * wid * 0.46f, yTail + 0.12f, -len / 2 - 0.01f), new Vector3(0.1f, 0.09f, 0.05f), IndOffMat(), default, false).GetComponent<Renderer>());
            }
        }

        static Material HeadOffMat() => M(new Color(0.85f, 0.85f, 0.8f), 0.9f);
        static Material TailOffMat() => M(new Color(0.4f, 0.04f, 0.04f), 0.7f);
        static Material IndOffMat() => M(new Color(0.55f, 0.35f, 0.05f), 0.7f);

        /// <summary>Indian plates: white = private, yellow = commercial</summary>
        static void Plates(Transform t, VehicleRig rig, int id, float len, float yFront, float yRear, bool commercial, bool front = true, float w = 0.52f)
        {
            rig.plate = PlateText(id);
            var pm = commercial ? M(new Color(0.96f, 0.8f, 0.1f), 0.3f) : M(new Color(0.95f, 0.95f, 0.95f), 0.3f);
            Shapes.Box(t, new Vector3(0, yRear, -len / 2 - 0.02f), new Vector3(w, 0.13f, 0.02f), pm, default, false);
            Shapes.Text(t, rig.plate, new Vector3(0, yRear, -len / 2 - 0.035f), 0.075f * w / 0.52f, Color.black);
            if (!front) return;
            Shapes.Box(t, new Vector3(0, yFront, len / 2 + 0.02f), new Vector3(w, 0.13f, 0.02f), pm, default, false);
            Shapes.Text(t, rig.plate, new Vector3(0, yFront, len / 2 + 0.035f), 0.075f * w / 0.52f, Color.black, new Vector3(0, 180, 0));
        }

        static void Car(Vehicle v, Transform t, VehicleRig rig)
        {
            var sp = v.spec; float L = sp.len, W = sp.wid;
            bool kaaliPeeli = v.id % 9 == 0;   // Mumbai black-and-yellow taxi
            Color[] palette = { new Color(0.9f, 0.9f, 0.92f), new Color(0.08f, 0.08f, 0.1f), new Color(0.6f, 0.62f, 0.66f),
                                new Color(0.62f, 0.08f, 0.08f), new Color(0.12f, 0.22f, 0.5f), new Color(0.97f, 0.97f, 0.97f), new Color(0.42f, 0.4f, 0.36f) };
            var paint = M(kaaliPeeli ? new Color(0.06f, 0.06f, 0.06f) : palette[v.id % palette.Length], 0.88f, v.id % 2 == 0 ? 0.45f : 0.1f);   // glossy, some metallic
            var roofPaint = kaaliPeeli ? M(new Color(0.98f, 0.8f, 0.05f), 0.6f) : paint;
            VehicleMesh.Part(t, VehicleMesh.CarBody(L, W), paint);
            VehicleMesh.Part(t, VehicleMesh.CarGlass(L, W), Glass, "Glass");
            VehicleMesh.Part(t, VehicleMesh.CarRoof(L, W), roofPaint, "Roof");
            Shapes.Box(t, new Vector3(0, 0.36f, L / 2 - 0.02f), new Vector3(W * 0.8f, 0.14f, 0.1f), Black);    // bumper lips
            Shapes.Box(t, new Vector3(0, 0.38f, -L / 2 + 0.02f), new Vector3(W * 0.78f, 0.14f, 0.1f), Black);
            Shapes.Box(t, new Vector3(0, 0.5f, L / 2 - 0.01f), new Vector3(W * 0.42f, 0.12f, 0.04f), Black, default, false);   // grille
            foreach (float s in new[] { -1f, 1f })
                Shapes.Box(t, new Vector3(s * (W / 2 + 0.06f), 1.0f, L * 0.19f), new Vector3(0.1f, 0.09f, 0.16f), paint, default, false);   // mirrors
            foreach (float x in new[] { -1f, 1f }) foreach (float z in new[] { -1f, 1f }) Wheel(t, x * (W / 2 - 0.16f), z * L * 0.32f, 0.32f, 0.22f, rig);
            Lights(t, rig, W, L - 0.06f, 0.6f, 0.72f);
            Plates(t, rig, v.id, L, 0.46f, 0.56f, kaaliPeeli);
        }

        static void Bike(Vehicle v, Transform t, VehicleRig rig, bool bicycle)
        {
            var sp = v.spec;
            if (bicycle)
            {
                var frame = M(new Color(0.15f, 0.3f, 0.6f), 0.5f, 0.3f);
                Wheel(t, 0, 0.55f, 0.33f, 0.04f, rig); Wheel(t, 0, -0.55f, 0.33f, 0.04f, rig);
                Shapes.Box(t, new Vector3(0, 0.6f, 0), new Vector3(0.05f, 0.05f, 1.0f), frame, new Vector3(-10, 0, 0));
                Shapes.Box(t, new Vector3(0, 0.75f, -0.1f), new Vector3(0.05f, 0.5f, 0.05f), frame);
                Shapes.Box(t, new Vector3(0, 1.0f, 0.45f), new Vector3(0.5f, 0.04f, 0.04f), Black);
            }
            else
            {
                Color[] palette = { Color.black, new Color(0.6f, 0.05f, 0.05f), new Color(0.1f, 0.2f, 0.5f), new Color(0.75f, 0.75f, 0.78f), new Color(0.2f, 0.2f, 0.22f) };
                var paint = M(palette[v.id % palette.Length], 0.7f, 0.2f);
                Wheel(t, 0, 0.68f, 0.3f, 0.1f, rig); Wheel(t, 0, -0.68f, 0.3f, 0.1f, rig);
                Shapes.Box(t, new Vector3(0, 0.6f, 0), new Vector3(0.22f, 0.32f, 1.15f), paint);
                Shapes.Box(t, new Vector3(0, 0.85f, 0.25f), new Vector3(0.3f, 0.2f, 0.45f), paint);
                Shapes.Box(t, new Vector3(0, 0.88f, -0.25f), new Vector3(0.28f, 0.1f, 0.6f), Black);
                Shapes.Box(t, new Vector3(0, 1.05f, 0.58f), new Vector3(0.7f, 0.05f, 0.05f), Black);
                rig.head.Add(Shapes.Box(t, new Vector3(0, 0.95f, 0.78f), new Vector3(0.16f, 0.14f, 0.06f), HeadOffMat(), default, false).GetComponent<Renderer>());
                rig.brake.Add(Shapes.Box(t, new Vector3(0, 0.78f, -0.88f), new Vector3(0.14f, 0.08f, 0.05f), TailOffMat(), default, false).GetComponent<Renderer>());
                rig.indL.Add(Shapes.Box(t, new Vector3(-0.18f, 0.92f, 0.72f), new Vector3(0.06f, 0.05f, 0.05f), IndOffMat(), default, false).GetComponent<Renderer>());
                rig.indR.Add(Shapes.Box(t, new Vector3(0.18f, 0.92f, 0.72f), new Vector3(0.06f, 0.05f, 0.05f), IndOffMat(), default, false).GetComponent<Renderer>());
                Plates(t, rig, v.id, 1.84f, 0f, 0.66f, false, false, 0.24f);
            }
            Rider(t, v.id, bicycle ? -0.15f : -0.1f, !bicycle && v.id % 4 != 0);
            if (!bicycle && v.id % 4 == 1) Rider(t, v.id + 3, -0.6f, v.id % 8 == 1, pillion: true);   // pillion, often without a helmet
        }

        static void Rider(Transform t, int id, float z, bool helmet, bool pillion = false)
        {
            float y = pillion ? -0.05f : 0f;
            Shapes.Part(t, PrimitiveType.Capsule, new Vector3(0, 1.32f + y, z), new Vector3(0.42f, 0.36f, 0.3f), Cloth(id));
            foreach (float s in new[] { -1f, 1f })
                Shapes.Box(t, new Vector3(s * 0.15f, 0.85f, z + 0.15f), new Vector3(0.13f, 0.14f, 0.5f), M(new Color(0.15f, 0.15f, 0.2f)), new Vector3(20, 0, 0));
            if (!pillion) Shapes.Box(t, new Vector3(0, 1.3f, z + 0.35f), new Vector3(0.55f, 0.1f, 0.45f), Cloth(id), new Vector3(30, 0, 0));
            Shapes.Part(t, PrimitiveType.Sphere, new Vector3(0, 1.8f + y, z + 0.02f), Vector3.one * 0.24f, Skin(id));
            if (helmet)
                Shapes.Part(t, PrimitiveType.Sphere, new Vector3(0, 1.84f + y, z), new Vector3(0.3f, 0.3f, 0.33f),
                            M(new[] { Color.black, new Color(0.85f, 0.1f, 0.1f), Color.white, new Color(0.1f, 0.2f, 0.6f) }[id % 4], 0.8f));
            else
                Shapes.Part(t, PrimitiveType.Sphere, new Vector3(0, 1.88f + y, z - 0.02f), new Vector3(0.25f, 0.14f, 0.26f), M(new Color(0.05f, 0.04f, 0.03f)));
        }

        static void Auto(Vehicle v, Transform t, VehicleRig rig)
        {
            var green = M(new Color(0.1f, 0.45f, 0.18f), 0.8f);
            var yellow = M(new Color(0.98f, 0.82f, 0.08f), 0.5f);
            Wheel(t, 0, 1.05f, 0.23f, 0.14f, rig); Wheel(t, -0.6f, -0.85f, 0.23f, 0.14f, rig); Wheel(t, 0.6f, -0.85f, 0.23f, 0.14f, rig);
            Shapes.Box(t, new Vector3(0, 0.36f, 0), new Vector3(1.2f, 0.08f, 2.5f), Black);
            VehicleMesh.Part(t, VehicleMesh.AutoBody(), green);
            Shapes.Box(t, new Vector3(0, 1.32f, 0.9f), new Vector3(1.1f, 0.6f, 0.04f), Glass, new Vector3(-10, 0, 0));
            VehicleMesh.Part(t, VehicleMesh.AutoCanopy(), yellow, "Canopy");
            foreach (float s in new[] { -1f, 1f })
            {
                Shapes.Box(t, new Vector3(s * 0.7f, 1.42f, -0.75f), new Vector3(0.04f, 0.68f, 1.1f), Black);
                Shapes.Box(t, new Vector3(s * 0.66f, 1.35f, 0.75f), new Vector3(0.05f, 0.8f, 0.05f), Black);
            }
            Shapes.Box(t, new Vector3(0, 1.0f, -0.65f), new Vector3(1.2f, 0.15f, 0.6f), M(new Color(0.15f, 0.15f, 0.15f)));
            Shapes.Part(t, PrimitiveType.Capsule, new Vector3(0, 1.2f, 0.45f), new Vector3(0.4f, 0.3f, 0.3f), Cloth(v.id + 2));
            Shapes.Part(t, PrimitiveType.Sphere, new Vector3(0, 1.55f, 0.5f), Vector3.one * 0.23f, Skin(v.id));
            rig.head.Add(Shapes.Box(t, new Vector3(0, 0.98f, 1.31f), new Vector3(0.2f, 0.16f, 0.05f), HeadOffMat(), default, false).GetComponent<Renderer>());
            foreach (float s in new[] { -1f, 1f })
            {
                rig.brake.Add(Shapes.Box(t, new Vector3(s * 0.5f, 0.72f, -1.26f), new Vector3(0.16f, 0.12f, 0.04f), TailOffMat(), default, false).GetComponent<Renderer>());
                (s < 0 ? rig.indL : rig.indR).Add(Shapes.Box(t, new Vector3(s * 0.3f, 1.1f, 1.31f), new Vector3(0.08f, 0.07f, 0.05f), IndOffMat(), default, false).GetComponent<Renderer>());
            }
            Plates(t, rig, v.id, 2.52f, 0.7f, 0.62f, true, true, 0.36f);
        }

        static void Bus(Vehicle v, Transform t, VehicleRig rig)
        {
            var sp = v.spec; float L = sp.len;
            var lower = M(v.id % 2 == 0 ? new Color(0.72f, 0.1f, 0.08f) : new Color(0.08f, 0.26f, 0.6f), 0.75f);
            var upper = M(new Color(0.93f, 0.9f, 0.82f), 0.4f);
            VehicleMesh.Part(t, VehicleMesh.BusPart(L, 0.4f, 1.72f, "lo"), lower);
            VehicleMesh.Part(t, VehicleMesh.BusPart(L, 1.72f, 3.1f, "hi"), upper, "Upper");
            Shapes.Box(t, new Vector3(0, 3.16f, -L * 0.1f), new Vector3(1.6f, 0.14f, L * 0.4f), M(new Color(0.55f, 0.55f, 0.55f)));   // roof unit
            foreach (float s in new[] { -1f, 1f }) Shapes.Box(t, new Vector3(s * 1.255f, 2.3f, -0.4f), new Vector3(0.02f, 0.85f, L - 1.8f), Glass, default, false);
            Shapes.Box(t, new Vector3(0, 2.15f, L / 2 + 0.01f), new Vector3(2.3f, 1.35f, 0.02f), Glass, default, false);
            Shapes.Box(t, new Vector3(-1.26f, 1.5f, L * 0.36f), new Vector3(0.03f, 2.3f, 1.0f), M(new Color(0.2f, 0.2f, 0.22f)), default, false);   // door, kerb side
            Shapes.Box(t, new Vector3(0, 2.95f, L / 2 + 0.02f), new Vector3(1.9f, 0.32f, 0.03f), Black, default, false);
            Shapes.Text(t, Routes[v.id % Routes.Length], new Vector3(0, 2.95f, L / 2 + 0.045f), 0.2f, new Color(1f, 0.55f, 0.1f), new Vector3(0, 180, 0));
            foreach (float x in new[] { -1f, 1f }) foreach (float z in new[] { 0.33f, -0.27f }) Wheel(t, x * 1.05f, z * L, 0.5f, 0.3f, rig);
            Lights(t, rig, 2.5f, L, 0.85f, 0.9f, 0.38f);
            Plates(t, rig, v.id, L, 0.55f, 0.6f, true);
        }

        static void Truck(Vehicle v, Transform t, VehicleRig rig)
        {
            var sp = v.spec; float L = sp.len;
            var cab = M(v.id % 3 == 0 ? new Color(0.9f, 0.45f, 0.1f) : v.id % 3 == 1 ? new Color(0.15f, 0.3f, 0.65f) : new Color(0.75f, 0.1f, 0.1f), 0.5f);
            var tarp = M(new[] { new Color(0.85f, 0.7f, 0.15f), new Color(0.2f, 0.35f, 0.7f), new Color(0.25f, 0.45f, 0.25f) }[v.id % 3], 0.1f);
            Shapes.Box(t, new Vector3(0, 1.55f, L / 2 - 0.95f), new Vector3(2.4f, 2.0f, 1.9f), cab);
            Shapes.Box(t, new Vector3(0, 2.05f, L / 2 + 0.01f), new Vector3(2.2f, 0.8f, 0.03f), Glass, default, false);
            Shapes.Box(t, new Vector3(0, 2.75f, L / 2 - 0.9f), new Vector3(2.3f, 0.3f, 1.4f), cab);   // decorated cab hood
            Shapes.Box(t, new Vector3(0, 1.15f, -1.0f), new Vector3(2.5f, 1.0f, L - 2.1f), M(new Color(0.45f, 0.3f, 0.18f)));
            Shapes.Box(t, new Vector3(0, 2.25f, -1.0f), new Vector3(2.4f, 1.3f, L - 2.3f), tarp);
            Shapes.Box(t, new Vector3(0, 1.2f, -L / 2 - 0.02f), new Vector3(2.3f, 0.55f, 0.03f), M(new Color(0.98f, 0.85f, 0.1f)), default, false);
            Shapes.Text(t, "HORN OK PLEASE", new Vector3(0, 1.2f, -L / 2 - 0.04f), 0.26f, new Color(0.8f, 0.05f, 0.05f));
            foreach (float x in new[] { -1f, 1f }) foreach (float z in new[] { 0.35f, -0.22f, -0.36f }) Wheel(t, x * 1.05f, z * L, 0.5f, 0.3f, rig);
            Lights(t, rig, 2.5f, L, 0.95f, 0.75f, 0.38f);
            Plates(t, rig, v.id, L, 0.55f, 0.6f, true);
        }

        static void Ambulance(Vehicle v, Transform t, VehicleRig rig)
        {
            float L = v.spec.len;
            var white = M(new Color(0.95f, 0.95f, 0.95f), 0.85f);
            VehicleMesh.Part(t, VehicleMesh.VanBody(L), white);
            Shapes.Box(t, new Vector3(0, 1.55f, L / 2 - 0.42f), new Vector3(1.8f, 0.62f, 0.05f), Glass, new Vector3(-38, 0, 0), false);
            Shapes.Box(t, new Vector3(0, 1.05f, -0.1f), new Vector3(2.02f, 0.25f, L - 0.2f), M(new Color(0.85f, 0.05f, 0.05f), 0.5f), default, false);
            Shapes.Box(t, new Vector3(0, 1.75f, -0.4f), new Vector3(2.03f, 0.5f, 0.12f), M(new Color(0.85f, 0.05f, 0.05f)), default, false);   // cross bar
            Shapes.Box(t, new Vector3(0, 1.75f, -0.4f), new Vector3(2.03f, 0.12f, 0.5f), M(new Color(0.85f, 0.05f, 0.05f)), default, false);
            rig.beaconA = Shapes.Box(t, new Vector3(-0.45f, 2.33f, L / 2 - 1.1f), new Vector3(0.45f, 0.16f, 0.25f), Mats.Glow(Color.red, 5f), default, false).GetComponent<Renderer>();
            rig.beaconB = Shapes.Box(t, new Vector3(0.45f, 2.33f, L / 2 - 1.1f), new Vector3(0.45f, 0.16f, 0.25f), Mats.Glow(Color.blue, 5f), default, false).GetComponent<Renderer>();
            var tm = Shapes.Text(t, "AMBULANCE", new Vector3(0, 1.38f, L / 2 + 0.03f), 0.2f, new Color(0.85f, 0.05f, 0.05f), new Vector3(0, 180, 0));
            tm.transform.localScale = new Vector3(-1, 1, 1);   // mirrored, to be read in a rear-view mirror
            var lg = new GameObject("Beacon light");
            lg.transform.SetParent(t, false); lg.transform.localPosition = new Vector3(0, 2.7f, L / 2 - 1.1f);
            rig.beaconLight = lg.AddComponent<UnityEngine.Light>();
            rig.beaconLight.type = LightType.Point; rig.beaconLight.range = 14f; rig.beaconLight.color = Color.red;
            foreach (float x in new[] { -1f, 1f }) foreach (float z in new[] { 0.32f, -0.3f }) Wheel(t, x * 0.85f, z * L, 0.38f, 0.24f, rig);
            Lights(t, rig, 2.0f, L, 0.85f, 0.9f);
            Plates(t, rig, v.id, L, 0.5f, 0.55f, false);
        }

        static void Cattle(Vehicle v, Transform t)
        {
            var hide = M(new Color(0.9f, 0.87f, 0.8f), 0.1f);
            var dark = M(new Color(0.35f, 0.3f, 0.26f), 0.1f);
            Shapes.Part(t, PrimitiveType.Capsule, new Vector3(0, 1.05f, 0), new Vector3(0.85f, 0.9f, 0.8f), hide, new Vector3(90, 0, 0));
            Shapes.Part(t, PrimitiveType.Sphere, new Vector3(0, 1.45f, 0.5f), new Vector3(0.45f, 0.35f, 0.4f), hide);   // hump
            Shapes.Box(t, new Vector3(0, 1.25f, 1.15f), new Vector3(0.38f, 0.42f, 0.55f), hide, new Vector3(25, 0, 0));
            foreach (float s in new[] { -1f, 1f })
            {
                Shapes.Box(t, new Vector3(s * 0.15f, 1.55f, 1.05f), new Vector3(0.05f, 0.25f, 0.05f), dark, new Vector3(-20, 0, s * 25));   // horns
                foreach (float z in new[] { -0.55f, 0.55f }) Shapes.Box(t, new Vector3(s * 0.25f, 0.35f, z), new Vector3(0.14f, 0.7f, 0.14f), hide);
            }
            Shapes.Box(t, new Vector3(0, 0.95f, -0.95f), new Vector3(0.06f, 0.7f, 0.06f), dark, new Vector3(15, 0, 0));
        }
    }
}
