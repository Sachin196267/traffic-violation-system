# PRAVAH robustness - 2220 simulation runs

12 simulator assumptions x 6 scenarios x 10 seeds x controllers, 30 simulated minutes each. Paired comparison (identical arrivals per seed), 95 % confidence intervals; "better"/"worse" only when the interval excludes 0.

## 1. Baseline (the simulator's default assumptions)

| Scenario | Controller | Wait per person (s) | Wait per vehicle (s) | Worst wait (s) | Red-light jumps /h | Throughput (veh/h) |
|---|---|---|---|---|---|---|
| City balanced | Fixed 30 s | 40.5 ± 2.5 | 40.9 ± 1.6 | 193.2 ± 76.6 | 62.4 ± 6.6 | 2297 ± 203 |
| City balanced | Webster | 33.9 ± 2.8 | 32.9 ± 3.0 | 152.4 ± 26.3 | 29.4 ± 9.3 | 2386 ± 37 |
| City balanced | Adaptive v1 | 26.3 ± 2.3 | 26.1 ± 2.7 | 118.0 ± 9.8 | 19.0 ± 4.4 | 2384 ± 39 |
| City balanced | PRAVAH | 17.2 ± 1.5 | 23.7 ± 1.4 | 110.2 ± 7.9 | 25.2 ± 6.9 | 2395 ± 36 |
| Office rush N-S | Fixed 30 s | 85.4 ± 9.0 | 84.0 ± 8.1 | 269.3 ± 27.3 | 117.2 ± 14.6 | 2345 ± 46 |
| Office rush N-S | Webster | 33.7 ± 4.2 | 33.4 ± 4.7 | 190.2 ± 40.8 | 35.2 ± 9.0 | 2423 ± 43 |
| Office rush N-S | Adaptive v1 | 22.3 ± 1.9 | 22.4 ± 1.9 | 125.0 ± 10.4 | 22.0 ± 5.5 | 2446 ± 44 |
| Office rush N-S | PRAVAH | 18.3 ± 1.3 | 23.2 ± 1.0 | 122.4 ± 8.1 | 25.4 ± 4.4 | 2443 ± 44 |
| Market road | Fixed 30 s | 43.5 ± 2.2 | 42.8 ± 2.5 | 163.7 ± 30.4 | 61.8 ± 8.5 | 2243 ± 36 |
| Market road | Webster | 30.2 ± 2.5 | 30.0 ± 1.7 | 151.4 ± 23.0 | 26.4 ± 8.2 | 2236 ± 37 |
| Market road | Adaptive v1 | 22.2 ± 1.6 | 21.9 ± 1.4 | 111.8 ± 10.6 | 18.0 ± 6.6 | 2251 ± 33 |
| Market road | PRAVAH | 16.1 ± 1.1 | 22.0 ± 1.3 | 115.1 ± 9.5 | 20.4 ± 5.1 | 2244 ± 34 |
| Peak near capacity | Fixed 30 s | 85.8 ± 11.1 | 82.8 ± 10.8 | 308.9 ± 50.4 | 145.4 ± 26.9 | 3060 ± 40 |
| Peak near capacity | Webster | 85.0 ± 11.2 | 84.5 ± 12.5 | 278.6 ± 53.3 | 161.4 ± 49.0 | 2765 ± 556 |
| Peak near capacity | Adaptive v1 | 66.0 ± 14.0 | 64.9 ± 12.1 | 242.4 ± 46.2 | 94.8 ± 27.1 | 2810 ± 660 |
| Peak near capacity | PRAVAH | 68.0 ± 7.9 | 68.2 ± 7.4 | 234.1 ± 22.9 | 114.6 ± 13.5 | 3073 ± 52 |
| Night | Fixed 30 s | 29.8 ± 3.8 | 31.4 ± 2.1 | 105.9 ± 0.9 | 29.6 ± 6.8 | 665 ± 18 |
| Night | Webster | 12.2 ± 0.8 | 12.0 ± 0.5 | 49.4 ± 5.6 | 0.4 ± 0.6 | 666 ± 18 |
| Night | Adaptive v1 | 5.4 ± 1.0 | 6.5 ± 0.6 | 67.1 ± 7.8 | 0.4 ± 0.6 | 667 ± 17 |
| Night | PRAVAH | 3.2 ± 0.5 | 4.9 ± 0.5 | 55.7 ± 8.8 | 0.0 ± 0.0 | 666 ± 18 |
| Ambulance corridor | Fixed 30 s | 41.9 ± 3.0 | 40.8 ± 1.8 | 159.6 ± 20.8 | 60.8 ± 11.2 | 2197 ± 42 |
| Ambulance corridor | Webster | 28.8 ± 2.6 | 28.0 ± 2.6 | 135.0 ± 18.8 | 21.6 ± 6.7 | 2194 ± 44 |
| Ambulance corridor | Adaptive v1 | 21.6 ± 1.7 | 21.3 ± 1.9 | 106.1 ± 13.4 | 16.2 ± 4.5 | 2205 ± 40 |
| Ambulance corridor | PRAVAH | 15.8 ± 1.0 | 20.9 ± 1.2 | 111.2 ± 11.0 | 21.2 ± 4.4 | 2204 ± 41 |

## 2. PRAVAH against each controller, over every assumption and scenario

Cells = (assumption, scenario) pairs. Better / no clear difference / worse for PRAVAH.

| Against | wait per person | wait per vehicle | worst wait | red-light jumps | left queued | throughput |
|---|---|---|---|---|---|---|
| Fixed 30 s | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 | 4 / 2 / 0 | 1 / 5 / 0 |
| Webster | 69 / 3 / 0 | 70 / 2 / 0 | 55 / 17 / 0 | 18 / 54 / 0 | 51 / 21 / 0 | 10 / 62 / 0 |
| Adaptive v1 | 60 / 12 / 0 | 17 / 51 / 4 | 4 / 64 / 4 | 1 / 60 / 11 | 2 / 69 / 1 | 2 / 69 / 1 |

## 3. Every case where PRAVAH is significantly worse

| Against | Metric | Assumption | Scenario | PRAVAH worse by (95 % CI) |
|---|---|---|---|---|
| Adaptive v1 | wait per vehicle | no free left turn | Office rush N-S | 3.0 ± 2.5 s |
| Adaptive v1 | wait per vehicle | no free left turn | Market road | 3.3 ± 1.9 s |
| Adaptive v1 | wait per vehicle | no free left turn | Ambulance corridor | 3.0 ± 1.3 s |
| Adaptive v1 | wait per vehicle | two-wheeler heavy mix (58%) | Office rush N-S | 1.5 ± 1.4 s |
| Adaptive v1 | worst wait | demand +20% | Ambulance corridor | 10.3 ± 8.4 s |
| Adaptive v1 | worst wait | demand -20% | Ambulance corridor | 10.9 ± 8.5 s |
| Adaptive v1 | worst wait | patient drivers (patience x1.5) | Office rush N-S | 11.0 ± 4.7 s |
| Adaptive v1 | worst wait | short camera range (50 m) | Ambulance corridor | 11.3 ± 11.1 s |
| Adaptive v1 | left queued | baseline | Market road | 3.5 ± 2.2 veh |
| Adaptive v1 | red-light jumps | demand +20% | Ambulance corridor | 15.0 ± 8.8 /h |
| Adaptive v1 | red-light jumps | demand -20% | Ambulance corridor | 4.6 ± 4.3 /h |
| Adaptive v1 | red-light jumps | long camera range (150 m) | City balanced | 8.8 ± 8.1 /h |
| Adaptive v1 | red-light jumps | no free left turn | City balanced | 10.0 ± 9.9 /h |
| Adaptive v1 | red-light jumps | no free left turn | Market road | 15.8 ± 9.8 /h |
| Adaptive v1 | red-light jumps | no free left turn | Ambulance corridor | 17.4 ± 9.4 /h |
| Adaptive v1 | red-light jumps | patient drivers (patience x1.5) | Office rush N-S | 4.4 ± 4.2 /h |
| Adaptive v1 | red-light jumps | short bus stops (dwell x0.5) | City balanced | 12.6 ± 8.6 /h |
| Adaptive v1 | red-light jumps | short bus stops (dwell x0.5) | Ambulance corridor | 6.4 ± 3.8 /h |
| Adaptive v1 | red-light jumps | short camera range (50 m) | City balanced | 10.0 ± 4.0 /h |
| Adaptive v1 | red-light jumps | two-wheeler heavy mix (58%) | City balanced | 9.0 ± 5.7 /h |
| Adaptive v1 | throughput | baseline | Market road | 6.6 ± 5.2 veh/h |

## 4. Does the main result survive each assumption change? (wait per person, mean over scenarios)

| Assumption | PRAVAH (s) | Best other (s) | Best other | PRAVAH better in scenarios |
|---|---|---|---|---|
| baseline | 23.1 | 27.3 | Adaptive v1 | 5 / 6 |
| impatient drivers (patience x0.5) | 19.3 | 24.0 | Adaptive v1 | 5 / 6 |
| patient drivers (patience x1.5) | 24.7 | 28.9 | Adaptive v1 | 5 / 6 |
| short bus stops (dwell x0.5) | 18.1 | 23.2 | Adaptive v1 | 6 / 6 |
| long bus stops (dwell x2) | 48.9 | 56.5 | Adaptive v1 | 3 / 6 |
| short camera range (50 m) | 24.0 | 28.3 | Adaptive v1 | 5 / 6 |
| long camera range (150 m) | 23.1 | 28.1 | Adaptive v1 | 5 / 6 |
| demand -20% | 10.9 | 15.6 | Adaptive v1 | 6 / 6 |
| demand +20% | 51.0 | 55.2 | Adaptive v1 | 5 / 6 |
| two-wheeler heavy mix (58%) | 15.7 | 20.3 | Adaptive v1 | 6 / 6 |
| no red-light jumping | 24.9 | 28.8 | Adaptive v1 | 5 / 6 |
| no free left turn | 30.6 | 34.7 | Adaptive v1 | 4 / 6 |
