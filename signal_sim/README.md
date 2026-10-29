# Smart Signal Simulator (India)

Unity 6 (6000.6.4f1). Open `signal_sim/` in Unity Hub and press Play - the junction, traffic and dashboard build
themselves in any scene. Headless comparison of all controllers:

    Unity -batchmode -nographics -projectPath signal_sim -executeMethod SignalSim.Experiment.Run [-only 2,3]
    -> Results/compare.md, Results/compare.csv     (Snapshot.Run renders Results/snapshot_*.png)

## Views and controls (F1 in Play shows this too)
| Key | View |
|---|---|
| 1 Orbit | right-drag rotate, WASD / middle-drag pan, scroll zoom |
| 2 Free fly | right-drag look, WASD move, Q/E down/up, Shift fast |
| 3 Top-down | scroll zoom |
| 4 CCTV | junction cameras with detection boxes (what the controller counts); press 4 again for the next camera |
| 5 Control room | all four CCTV feeds |
| 6 Follow | click any vehicle (or F); V = driver's seat / chase cam |
| 7 Ambulance | chase the nearest ambulance |
| 8 Drone | cinematic orbit |

N day / evening / night, R monsoon rain, Space pause, H hide panel, M minimap. Snapshots of every view: `Results/snap_*.png`.

## Rendering
Universal Render Pipeline (set up automatically by `Editor/RenderSetup.cs`: linear colour, HDR, 4x MSAA, 4K soft shadows to
180 m, ambient occlusion), ACES tone mapping, bloom, grading and depth of field when following (`View/Post.cs`).
Photographed CC0 skies (HDRI) and scanned asphalt / pavers / plaster / concrete / ground from Poly Haven in
`Assets/Resources/PH` (`python3 Tools/fetch_assets.py` re-downloads them); the sun is placed where each sky's sun is.
Vehicle bodies are lofted curved meshes (`View/VehicleMesh.cs`); monsoon mode wets the roads and adds rain.

## Indian traffic modelled
Mixed traffic (38 % two-wheelers, autos, cars, buses, trucks, bicycles, ambulances), left-hand driving, lane-less
two-wheelers that filter and creep onto the zebra, red-light jumping by impatient riders/drivers, free left turn,
buses and autos stopping just past the junction (spillback into the box; a blocked box stops cross traffic),
stray cattle in the kerb lane, IRC:106 PCU values, IRC:SP:41 saturation flow (525 x width), cameras that see 100 m.

## Controllers
| | |
|---|---|
| Fixed 30 s | naive rotation |
| Webster | fixed-time plan designed from the counted flows (Webster / IRC:SP:41) |
| Adaptive v1 | C# re-implementation inspired by `modules/smart_signal_controller.py` (PCU queue pressure, ambulance preemption) |
| Python bridge | the signal decided by Python: the real `modules/smart_signal_controller.py` or PRAVAH's Python port - see `signal_bridge/` and `Results/bridge.md` |
| PRAVAH | new algorithm, below |

## PRAVAH - Person-weighted, Red-patience-Aware, Vehicle-mix Adaptive Headway control
1. **People, not vehicles** - each waiting vehicle counts its occupants (bus ~40, auto 2.5, bike 1.4).
2. **Patience-aware cost** - a person who has waited w s costs 1 + (w/75)^2 per second; past ~90 s of red
   (where jumping takes off) that approach is served first, so a full bus can never starve a few bikes.
3. **Self-calibrating discharge** - learns each approach's real saturation flow and arrival rate from the camera,
   starting from IRC:SP:41.
4. **Spillback guard** - discounts approaches whose exit arm is full; ends a green that would feed a blocked box.
5. **Queue-flush ambulance timing** - switches when the ambulance's ETA equals the time needed to clear the queue
   ahead of it, instead of stopping the junction the moment it is seen.
6. **Marginal relief rule** - after clearing its planned queue, a green continues only while it relieves more
   person-urgency per second than the best other approach would (switching loss included).

Source: `Assets/Scripts/Sim/Controllers.cs` (`PravahController`). Simulation: `SimCore.cs`. View: `SimView.cs`.
