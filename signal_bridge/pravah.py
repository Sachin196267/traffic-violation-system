"""
PRAVAH - Person-weighted, Red-patience-Aware, Vehicle-mix Adaptive Headway signal control.
Python port of PravahController in signal_sim/Assets/Scripts/Sim/Controllers.cs (same rules, same constants).

Input each call = what the junction cameras report (see server.py for the message):
  app[a]  per-approach counters, exit_occ[b], box, veh = visible waiting vehicles.
"""
NAMES = ["North", "East", "South", "West"]
IRC_SAT_FLOW = 525 * 7.0 / 3600   # IRC:SP:41: 525 x approach width (7 m) PCU/h -> PCU/s
AMBER, ALL_RED = 3.0, 2.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Pravah:
    name = "PRAVAH (Python)"

    def __init__(self, tau=75.0, min_green=6.0, max_green=50.0, jump_risk=90.0):
        self.tau, self.min_green, self.max_green, self.jump_risk = tau, min_green, max_green, jump_risk
        self.lost = AMBER + ALL_RED
        self.sat = [IRC_SAT_FLOW] * 4     # learned discharge, PCU/s
        self.arrival = [0.0] * 4          # learned arrivals into the camera zone, PCU/s
        self.last_arr = [0.0] * 4
        self.last_t, self.win_app, self.win_crossed, self.win_queue = -1.0, -1, 0.0, 0.0

    # ---------- helpers ----------
    def urg(self, w):
        return 1.0 + (w / self.tau) ** 2

    def measure(self, m, a, moving=False):
        """waiting vehicles of approach a; moving=True adds the ones already rolling to the line (green approach)"""
        n = pcu = persons = urg = occ_w = 0.0
        for v in m["veh"]:
            if v["a"] != a:
                continue
            if (v["dist"] > 60.0) if moving else (v["v"] >= 2.0):
                continue
            n += 1; pcu += v["pcu"]; persons += v["persons"]; urg += v["persons"] * self.urg(v["wait"])
            occ_w += m["exit_occ"][v["exit"]] * v["pcu"]
        occ = occ_w / pcu if pcu > 0 else m["exit_occ"][(a + 2) % 4]
        return {"n": int(n), "pcu": pcu, "persons": persons, "urg": urg, "occ": occ}

    @staticmethod
    def spill(occ):
        return clamp(1.0 - (occ - 0.5) / 0.4, 0.0, 1.0)

    def clear(self, a, pcu):
        return 2.0 + pcu / self.sat[a]

    def gain(self, m, a):
        q = self.measure(m, a)
        if q["n"] == 0:
            near = m["app"][a]["near"]
            return (0.01 * near if near > 0 else 0.0), q
        return q["urg"] * self.spill(q["occ"]) / (self.clear(a, q["pcu"]) + self.lost), q

    def learn(self, m):
        app, t = m["app"], m["t"]
        if self.last_t < 0:
            self.last_t = t
            self.last_arr = [app[a]["arrived_pcu"] for a in range(4)]
            return
        dt = t - self.last_t
        if dt < 1.0:
            return
        for a in range(4):
            rate = (app[a]["arrived_pcu"] - self.last_arr[a]) / dt
            self.arrival[a] += (rate - self.arrival[a]) * min(1.0, 0.02 * dt)
            self.last_arr[a] = app[a]["arrived_pcu"]
        g = m["green"]
        green = m["phase"] == "Green"
        if self.win_app == g and self.win_queue >= 3.0 and green and dt < 1.5:
            rate = (app[g]["crossed_pcu"] - self.win_crossed) / dt
            self.sat[g] = clamp(self.sat[g] + (rate - self.sat[g]) * 0.1, 0.3, 3.0)
        ok = green and m["phase_t"] > 3.0
        self.win_app = g if ok else -1
        self.win_crossed = app[g]["crossed_pcu"]
        self.win_queue = app[g]["pcu"] if ok else 0.0
        self.last_t = t

    @staticmethod
    def ambulance(m):
        best = -1
        for a in range(4):
            s = m["app"][a]
            if s["amb"] and (best < 0 or s["amb_dist"] < m["app"][best]["amb_dist"]):
                best = a
        return best

    # ---------- decisions ----------
    def next(self, m):
        amb = self.ambulance(m)
        if amb >= 0:
            qa = self.measure(m, amb)
            g = clamp(self.clear(amb, qa["pcu"]) + 6.0, 10.0, self.max_green)
            return amb, g, f"PRAVAH (Python): ambulance on {NAMES[amb]} {m['app'][amb]['amb_dist']:.0f} m out - flushing {qa['n']} vehicles ahead of it"
        late = -1
        for a in range(4):
            if a != m["green"] and m["app"][a]["max_wait"] > self.jump_risk and (late < 0 or m["app"][a]["max_wait"] > m["app"][late]["max_wait"]):
                late = a
        best, best_gain, bq = -1, 0.0, None
        for a in range(4):
            if late >= 0 and a != late:
                continue
            gn, q = self.gain(m, a)
            if a == late:
                gn = max(gn, 1e-3)
            if a == m["green"] and m["t"] > 1.0:
                gn *= 0.5
            if gn > best_gain:
                best, best_gain, bq = a, gn, q
        if best < 0:
            return (m["green"] + 1) % 4, self.min_green, "PRAVAH (Python): no one waiting - short green, round robin"
        c = self.clear(best, bq["pcu"])
        g = clamp(c + self.arrival[best] * c / self.sat[best], self.min_green, self.max_green)
        return best, g, (f"PRAVAH (Python): {NAMES[best]} - {bq['n']} veh, ~{bq['persons']:.0f} people, urgency {bq['urg']:.0f}; "
                         f"clears in ~{c:.0f} s (learned {self.sat[best] * 3600:.0f} PCU/h), exit {bq['occ'] * 100:.0f}% full")

    def hold(self, m):
        self.learn(m)
        a, el = m["green"], m["phase_t"]
        amb = self.ambulance(m)
        if amb == a and el < self.max_green + 30.0:
            return True, ""
        if amb >= 0 and el >= self.min_green:
            s = m["app"][amb]
            eta = s["amb_dist"] / max(s["amb_speed"], 6.0)
            need = self.clear(amb, self.measure(m, amb)["pcu"]) + self.lost
            if eta <= need + 2.0:
                return False, f"queue-flush: ambulance on {NAMES[amb]} arrives in {eta:.0f} s, its queue needs {need:.0f} s"
        if el < self.min_green:
            return True, ""
        q = self.measure(m, a, moving=True)
        if self.spill(q["occ"]) < 0.15 and m["box"] != 0:
            return False, f"spillback: {NAMES[a]}'s exit is full - stop feeding the box"
        if q["n"] == 0:
            return False, "gap-out: approach empty"
        if el >= self.max_green:
            return False, "maximum green reached"
        if el < m["green_dur"]:
            return True, ""
        for o in range(4):
            if o != a and m["app"][o]["max_wait"] > self.jump_risk:
                return False, f"patience: {NAMES[o]} has waited {m['app'][o]['max_wait']:.0f} s"
        relief = self.sat[a] * (q["urg"] / max(q["pcu"], 0.1)) * self.spill(q["occ"])
        other, ob = 0.0, -1
        for o in range(4):
            if o == a:
                continue
            gn, _ = self.gain(m, o)
            if gn > other:
                other, ob = gn, o
        if relief >= other:
            return True, ""
        return False, f"switch: {NAMES[a]} relieves {relief:.1f}/s < {other:.1f}/s waiting on {NAMES[ob]}"
