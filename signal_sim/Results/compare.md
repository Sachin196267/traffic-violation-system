# Signal controllers on an Indian junction - 30 simulated minutes per run, seeds 1, 2, 3

Averages over seeds; every controller gets the same arrivals per seed. Wait = time stopped before the stop line; person wait weights each vehicle by its occupants. Ambulance wait averages only runs that had an ambulance.

| Scenario | Controller | Avg wait (s) | Person wait (s) | Worst wait (s) | Ambulance wait (s) | Red-light jumps /h | Box blocked (%) | Throughput (veh/h) | Queued at end |
|---|---|---|---|---|---|---|---|---|---|
| City balanced | Fixed 30 s | 40.2 | 38.9 | 176 | 39.0 | 70 | 13 | 2381 | 44 |
| City balanced | Webster | 35.6 | 36.3 | 156 | 40.5 | 33 | 8 | 2373 | 43 |
| City balanced | Adaptive v1 | 27.4 | 26.9 | 139 | 4.0 | 20 | 8 | 2375 | 37 |
| City balanced | PRAVAH | 22.3 | 15.5 | 125 | 0.1 | 23 | 5 | 2388 | 30 |
| Office rush N-S | Fixed 30 s | 86.7 | 89.8 | 288 | 85.8 | 115 | 15 | 2339 | 95 |
| Office rush N-S | Webster | 36.3 | 36.3 | 254 | 23.9 | 39 | 12 | 2388 | 51 |
| Office rush N-S | Adaptive v1 | 23.8 | 23.3 | 158 | 4.7 | 26 | 10 | 2441 | 30 |
| Office rush N-S | PRAVAH | 24.5 | 19.4 | 141 | 0.2 | 26 | 11 | 2434 | 29 |
| Market road | Fixed 30 s | 40.3 | 41.5 | 131 | 34.9 | 67 | 11 | 2241 | 41 |
| Market road | Webster | 31.0 | 30.7 | 168 | 28.1 | 23 | 7 | 2223 | 41 |
| Market road | Adaptive v1 | 21.9 | 22.3 | 122 | 0.5 | 24 | 5 | 2242 | 26 |
| Market road | PRAVAH | 20.4 | 14.4 | 134 | 0.1 | 18 | 5 | 2238 | 29 |
| Peak near capacity | Fixed 30 s | 84.5 | 87.0 | 399 | 73.0 | 141 | 22 | 3019 | 157 |
| Peak near capacity | Webster | 96.1 | 98.1 | 372 | 120.0 | 211 | 21 | 2986 | 172 |
| Peak near capacity | Adaptive v1 | 73.2 | 75.7 | 268 | 8.9 | 109 | 21 | 3063 | 125 |
| Peak near capacity | PRAVAH | 73.3 | 73.8 | 295 | 7.6 | 119 | 24 | 3010 | 148 |
| Night | Fixed 30 s | 33.5 | 31.6 | 107 | 24.8 | 28 | 0 | 653 | 5 |
| Night | Webster | 12.2 | 11.1 | 56 | 31.2 | 0 | 0 | 650 | 4 |
| Night | Adaptive v1 | 6.8 | 5.5 | 80 | 0.1 | 0 | 0 | 654 | 2 |
| Night | PRAVAH | 4.5 | 2.9 | 58 | 0.1 | 0 | 0 | 652 | 3 |
| Ambulance corridor | Fixed 30 s | 39.6 | 41.0 | 173 | 33.5 | 69 | 10 | 2201 | 33 |
| Ambulance corridor | Webster | 29.5 | 29.6 | 177 | 21.0 | 25 | 7 | 2171 | 37 |
| Ambulance corridor | Adaptive v1 | 19.8 | 20.2 | 101 | 0.8 | 15 | 6 | 2191 | 25 |
| Ambulance corridor | PRAVAH | 20.9 | 15.9 | 148 | 0.4 | 18 | 6 | 2199 | 23 |
