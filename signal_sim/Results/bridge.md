# Python in the loop (signal_bridge)

Unity runs the junction; the signal is decided by a Python process (`signal_bridge/server.py`).
30 simulated minutes per run, seeds 1-3, same arrivals for every controller.

## 1. Python PRAVAH is an exact port
Shadow test (`Editor/BridgeCheck.cs`): C# PRAVAH drives the signal; at every decision Python receives the identical
camera state. Disagreements: **0 of 83,702 decisions** (0 / 635 phase choices, 0 / 83,067 hold-or-switch) over all six
scenarios. A first run showed 0.1-1.2 % disagreement - traced to floats being rounded in the JSON (phase timer 5.9999995 s
sent as 6.0); fixed by sending full precision.

## 2. The project's own controller (modules/smart_signal_controller.py) in closed loop
The cameras' vehicles are passed to `SmartSignalController.update()` as `VehicleAssignment`s (lane = approach).

| Scenario | Avg wait (s) | Person wait (s) | Worst (s) | Ambulance (s) | Red jumps /h | Box blocked % | veh/h | Queued at end |
|---|---|---|---|---|---|---|---|---|
| City balanced | 29.8 | 26.5 | 220 | 8.7 | 37 | 5 | 2373 | 40 |
| Office rush N-S | 29.4 | 27.3 | 309 | 3.4 | 64 | 8 | 2432 | 33 |
| Market road | 27.6 | 25.1 | 181 | 0.1 | 50 | 7 | 2237 | 35 |
| Peak near capacity | 98.7 | 95.0 | 390 | 67.8 | 265 | 34 | 2339 | 486 |
| Night | 9.5 | 4.6 | 133 | 0.2 | 5 | 0 | 654 | 3 |
| Ambulance corridor | 23.1 | 20.9 | 169 | 1.5 | 19 | 4 | 2195 | 26 |

Compared with compare.md: better than Fixed 30 s and Webster in most scenarios, behind PRAVAH everywhere, and it breaks
down near capacity (throughput drops ~22 %, 486 vehicles left queued). Its projected-wait model has no fairness limit or
spillback guard and re-decides every 0.5 s, which near saturation produces long reds and red-light jumping.
Note: the C# "Adaptive v1" was a re-implementation inspired by this module; these numbers are the module itself.
