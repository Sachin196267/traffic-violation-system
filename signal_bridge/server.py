"""
Signal-control bridge: the Unity simulator (signal_sim) asks this process what the signal should do.

    python3 signal_bridge/server.py --controller pravah     # PRAVAH, Python port
    python3 signal_bridge/server.py --controller project    # modules/smart_signal_controller.py

Protocol: TCP 127.0.0.1:5055, one JSON object per line. On connect the server sends {"name": ...}.
Unity then sends {"op": "next" | "hold", "t", "green", "phase", "phase_t", "green_dur", "box",
                  "app": [4 x {pcu, persons, max_wait, near, amb, amb_dist, amb_speed, crossed_pcu, arrived_pcu}],
                  "exit_occ": [4], "veh": [{id, a, type, pcu, persons, wait, v, dist, exit, jumper}]}
and gets {"approach", "green", "reason"} for next, {"hold", "reason"} for hold.
Each connection is one simulation run with a fresh controller. Decisions are logged to signal_bridge/logs/.
"""
import argparse
import json
import socket
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make(kind):
    if kind == "project":
        from project_controller import ProjectController
        return ProjectController()
    from pravah import Pravah
    return Pravah()


def serve(kind, port):
    make(kind)   # import once up front so the first client does not wait for module loading
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(4)
    (HERE / "logs").mkdir(exist_ok=True)
    print(f"signal bridge: {kind} controller on 127.0.0.1:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ctl = make(kind)
        log = open(HERE / "logs" / f"decisions_{kind}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w")
        f = conn.makefile("rwb", buffering=65536)
        n = 0
        try:
            f.write((json.dumps({"name": ctl.name}) + "\n").encode()); f.flush()
            for line in f:
                m = json.loads(line)
                if m["op"] == "next":
                    a, g, why = ctl.next(m)
                    out = {"approach": int(a), "green": float(g), "reason": why}
                    log.write(json.dumps({"t": round(m["t"], 1), "green": int(a), "planned_s": round(g, 1), "reason": why}) + "\n")
                else:
                    keep, why = ctl.hold(m)
                    out = {"hold": bool(keep), "reason": why}
                    if not keep:
                        log.write(json.dumps({"t": round(m["t"], 1), "end_green": m["green"], "after_s": round(m["phase_t"], 1), "reason": why}) + "\n")
                f.write((json.dumps(out) + "\n").encode()); f.flush()
                n += 1
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            log.close(); conn.close()
            print(f"run finished: {n} decisions", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["pravah", "project"], default="pravah")
    ap.add_argument("--port", type=int, default=5055)
    a = ap.parse_args()
    serve(a.controller, a.port)
