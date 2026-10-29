#!/usr/bin/env python3
"""
Stage 2 - lane inference from traffic only (no paint, no edges, no colour).

Road model
----------
For a fixed camera looking at a flat road, a ground point at lateral offset X
from a (possibly curving) reference line projects to

        x = g(y) + u * (y - y_h)          with  u = X / camera_height

where y_h is the horizon row and g(y) is the image of the reference line.
So in the (u, y) coordinate system every lane line is a *vertical line*
(constant u) and every lane has the *same width* in u regardless of distance.
That is the core difference from clustering raw pixel x-positions, where
lanes shrink towards the horizon and bend with perspective.

Pipeline
--------
1. Trajectories -> ~1 s chunks of ground-contact points (bottom-centre of box).
2. y_h: from how box width grows with y (car width = k*(y - y_h)), refined by
   a scan minimising the trajectory fit residual.
3. Joint linear least squares (IRLS/Huber) for g(y) and one u per chunk.
4. Vehicle width in u-units gives the physical scale for free:
   lane width ~ 1.95 x car width (3.5 m lane / 1.8 m car).
5. Per carriageway (direction + empty gaps) fit an evenly spaced "comb" of
   lane centres to the u-histogram.  Weak comb => no lane discipline =>
   virtual lanes = road width / standard lane width.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

CAR, MOTO, BUS, TRUCK, BICYCLE = 2, 3, 5, 7, 1
LANE_PER_CAR = 1.95        # 3.5 m lane / ~1.8 m car (incl. mirrors & bbox slack)


# --------------------------------------------------------------------------- data

@dataclass
class Chunks:
    x: np.ndarray            # contact points (all chunks concatenated)
    y: np.ndarray
    cid: np.ndarray          # chunk index per point
    ctrack: np.ndarray       # track id per chunk
    cdir: np.ndarray         # +1 moving down (towards camera), -1 up
    cw: np.ndarray           # weight per chunk (each track sums to 1)
    cls: np.ndarray          # class per chunk
    n: int


def build_chunks(det: np.ndarray, W: int, H: int, chunk_s: float = 1.0,
                 min_pts: int = 4, min_disp_frac: float = 0.04, y_h: float | None = None,
                 car_u: float | None = None) -> Chunks:
    """det rows: t, tid, cls, x1, y1, x2, y2, conf

    Ground contact point.  The box bottom (y2) is the front/rear edge of the
    footprint, which is horizontal in the image (lateral = horizontal in our
    model).  On a diagonal road (dx/dy = s > 0) the box's right edge x2 is that
    edge's outer corner, so the edge centre is x2 - half the car's lateral
    width; mirrored for s < 0.  Half-width = car_u/2 * (y2 - y_h) grows
    linearly with depth, so an error in car_u shifts every lane equally.
    Without (y_h, car_u) yet, fall back to the box bottom-centre.
    """
    t, tid, cls, x1, y1, x2, y2 = (det[:, i] for i in range(7))
    # a box cut by the frame border has a wrong contact point
    keep = (y2 < H - 3) & (x1 > 2) & (x2 < W - 3) & (cls != 0)
    xs, ys, cids, ctr, cdir, cwt, ccl = [], [], [], [], [], [], []
    for k in np.unique(tid[keep]):
        m = keep & (tid == k)
        if m.sum() < min_pts:
            continue
        o = np.argsort(t[m])
        tt = t[m][o]
        cy = y2[m][o]
        bw = (x2[m] - x1[m])[o]
        cx = ((x1[m] + x2[m]) / 2)[o]
        # Ground contact: the lowest point of a vehicle is the end of its footprint
        # that points down the image.  When the road runs diagonally (dx/dy = s) that
        # end is shifted towards +sign(s), so slide along the box bottom accordingly.
        if y_h is not None and np.ptp(cy) > 5:
            slope = np.polyfit(cy, cx, 1)[0]
            hw = np.minimum(0.5 * car_u * np.maximum(cy - y_h, 1), 0.5 * bw)
            xl, xr = cx - bw / 2, cx + bw / 2
            a = np.clip(slope / 0.3, -1, 1)                # smooth hand-over around s = 0
            cx = np.where(a >= 0, a * (xr - hw) + (1 - a) * cx, -a * (xl + hw) + (1 + a) * cx)
        if len(cx) >= 3:                                   # 3-tap median de-jitter
            cx = np.median(np.stack([np.r_[cx[0], cx[:-1]], cx, np.r_[cx[1:], cx[-1]]]), 0)
            cy = np.median(np.stack([np.r_[cy[0], cy[:-1]], cy, np.r_[cy[1:], cy[-1]]]), 0)
        disp = np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])
        if disp < min_disp_frac * H:                       # parked / queued without moving
            continue
        d = 1 if cy[-1] > cy[0] else -1
        bins = ((tt - tt[0]) // chunk_s).astype(int)
        groups = [np.where(bins == b)[0] for b in np.unique(bins)]
        groups = [g for g in groups if len(g) >= min_pts]
        if not groups:
            continue
        w = 1.0 / len(groups)
        cl = int(np.bincount(cls[m].astype(int)).argmax())
        for g in groups:
            c = len(ctr)
            xs.append(cx[g]); ys.append(cy[g]); cids.append(np.full(len(g), c))
            ctr.append(k); cdir.append(d); cwt.append(w); ccl.append(cl)
    if not ctr:
        raise ValueError("no usable moving vehicle tracks")
    return Chunks(np.concatenate(xs), np.concatenate(ys), np.concatenate(cids),
                  np.array(ctr), np.array(cdir), np.array(cwt), np.array(ccl), len(ctr))


def horizon_from_widths(det: np.ndarray, W: int, H: int):
    """Box width of cars grows linearly with (y - y_h). Robust fit -> y_h."""
    cls, x1, y2, x2 = det[:, 2], det[:, 3], det[:, 6], det[:, 5]
    m = (cls == CAR) & (x1 > 2) & (x2 < W - 3) & (y2 < H - 3)
    if m.sum() < 30:
        m = np.isin(cls, [CAR, BUS, TRUCK]) & (x1 > 2) & (x2 < W - 3) & (y2 < H - 3)
    w, y = (x2 - x1)[m], y2[m]
    if len(w) < 20:
        return None
    wt = np.ones_like(w)
    for _ in range(10):                                    # IRLS, Huber
        A = np.c_[y, np.ones_like(y)] * wt[:, None]
        a, b = np.linalg.lstsq(A, w * wt, rcond=None)[0]
        r = w - (a * y + b)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        wt = np.sqrt(np.clip(1.5 * s / (np.abs(r) + 1e-9), 0, 1))
    if a <= 1e-4:
        return None
    return float(-b / a)


# --------------------------------------------------------------------------- geometry

@dataclass
class Geometry:
    y_h: float
    gcoef: list                # g(y) = sum gcoef[j] * (y - y_h) ** powers[j]
    powers: list
    y_top: float               # far limit of observed traffic
    y_bot: float
    resid_px: float

    def g(self, y):
        d = np.asarray(y, float) - self.y_h
        return sum(c * d ** p for c, p in zip(self.gcoef, self.powers))

    def x_of(self, u, y):
        return self.g(y) + u * (np.asarray(y, float) - self.y_h)

    def u_of(self, x, y):
        return (np.asarray(x, float) - self.g(y)) / (np.asarray(y, float) - self.y_h)


def fit_geometry(ch: Chunks, y_h: float, degree: int, iters: int = 30):
    """Linear model x = sum_j c_j (y-y_h)^p_j + u_chunk (y-y_h); IRLS with Huber.
    Solved by alternating closed forms (exact for this bilinear-free model)."""
    powers = [0] + list(range(2, degree + 1))
    d = ch.y - y_h
    valid = d > 1                                          # points above the horizon are outliers
    if valid.mean() < 0.9:
        return None
    d = np.maximum(d, 1)
    B = np.stack([d ** p for p in powers], 1)
    B /= np.abs(B).max(0)                                  # conditioning
    scale = np.stack([np.abs(d ** p).max() for p in powers])
    npts = np.bincount(ch.cid, minlength=ch.n)
    pw = (ch.cw / np.maximum(npts, 1))[ch.cid] * valid     # every track weighs the same
    rw = np.ones_like(ch.x)
    k = B.shape[1]
    for _ in range(iters):
        w = pw * rw
        # exact weighted least squares for (c, u) by eliminating the per-chunk u
        # (Schur complement); alternating updates converge far too slowly when
        # the horizon is close to the top of the image
        a = np.bincount(ch.cid, w * d * d, ch.n) + 1e-12
        bvec = np.stack([np.bincount(ch.cid, w * d * B[:, j], ch.n) for j in range(k)], 1)
        xk = np.bincount(ch.cid, w * d * ch.x, ch.n)
        Sm = (B * w[:, None]).T @ B - (bvec / a[:, None]).T @ bvec
        rhs = (B * w[:, None]).T @ ch.x - (bvec / a[:, None]).T @ xk
        c = np.linalg.lstsq(Sm + 1e-9 * np.eye(k), rhs, rcond=None)[0]
        u = (xk - bvec @ c) / a
        e = ch.x - B @ c - u[ch.cid] * d
        s = 1.4826 * np.median(np.abs(e)) + 1e-6
        rw = np.clip(2.0 * s / (np.abs(e) + 1e-9), 0, 1)
    resid = float(np.sum(pw * np.abs(e)) / np.sum(pw))
    coef = list(c / scale)
    return coef, powers, u, resid


def _track_residual(ch: Chunks, gcoef, powers, y_h, p0):
    """Per-chunk shape residual (in lane widths) when each chunk gets its own best u."""
    d = np.maximum(ch.y - y_h, 1)
    gx = sum(c * d ** p for c, p in zip(gcoef, powers))
    r = ch.x - gx
    u = np.bincount(ch.cid, r * d, ch.n) / (np.bincount(ch.cid, d * d, ch.n) + 1e-9)
    e = np.abs(r - u[ch.cid] * d) / (p0 * d)
    o = np.lexsort((e, ch.cid))                            # median per chunk, vectorised
    cnt = np.bincount(ch.cid, minlength=ch.n)
    start = np.r_[0, np.cumsum(cnt)[:-1]]
    res = np.full(ch.n, np.inf)
    ok = cnt > 0
    res[ok] = e[o][start[ok] + cnt[ok] // 2]
    return res, u


def subset(ch: Chunks, cmask: np.ndarray) -> Chunks:
    idx = np.where(cmask)[0]
    remap = -np.ones(ch.n, int)
    remap[idx] = np.arange(len(idx))
    pm = cmask[ch.cid]
    return Chunks(ch.x[pm], ch.y[pm], remap[ch.cid[pm]], ch.ctrack[idx], ch.cdir[idx], ch.cw[idx],
                  ch.cls[idx], len(idx))


def choose_degree(ch: Chunks, y_h: float, verbose=False):
    best = None
    n_tracks = len(np.unique(ch.ctrack))
    # curvature needs many long tracks to be trusted; otherwise a straight road
    degrees = (1, 2, 3) if n_tracks >= 25 else (1,)
    for degree in degrees:
        f = fit_geometry(ch, y_h, degree)
        if f is None:
            continue
        if verbose:
            print(f"      degree {degree}: resid={f[3]:.2f}px")
        if best is None or f[3] < 0.7 * best[1][3]:        # higher order must clearly pay off
            best = (degree, f)
    return best


def _merge_cost(tr, a, b, y_h):
    """Joint-fit residual relative to separate fits (<1.35 => same road geometry)."""
    fb = choose_degree(subset(tr, a | b), y_h)
    fa, fm = choose_degree(subset(tr, a), y_h), choose_degree(subset(tr, b), y_h)
    if fb is None or fa is None or fm is None:
        return np.inf
    sep = (fa[1][3] * a.sum() + fm[1][3] * b.sum()) / (a.sum() + b.sum())
    return fb[1][3] / max(sep, 1e-6)


def find_roads_by_direction(tr: Chunks, y_h: float, p0: float, rng, verbose=False):
    """Opposite carriageways are often parallel (same geometry) but not always
    (service roads, flyover ramps).  Find roads per direction, then merge
    across directions only when a joint fit is as good as separate ones."""
    roads = []
    for d in (1, -1):
        m = tr.cdir == d
        if m.sum() >= 5:
            roads += find_roads(tr, y_h, p0, rng, init=m, verbose=verbose)
    merged = True
    while merged and len(roads) > 1:
        merged = False
        for i in range(len(roads)):
            for j in range(i + 1, len(roads)):
                if _merge_cost(tr, roads[i], roads[j], y_h) < 1.35:
                    roads[i] = roads[i] | roads[j]
                    roads.pop(j)
                    merged = True
                    break
            if merged:
                break
    return roads


def find_roads(tr: Chunks, y_h: float, p0: float, rng, max_roads=3, inlier_lane=0.12, verbose=False,
               init=None):
    """Sequential RANSAC over whole tracks: each road = one smooth g(y) that all its
    tracks follow at some constant lateral offset u."""
    remaining = np.ones(tr.n, bool) if init is None else init.copy()
    n_pool = int(remaining.sum())
    span = np.array([np.ptp(tr.y[tr.cid == c]) for c in range(tr.n)])
    roads = []
    for _ in range(max_roads):
        idx = np.where(remaining)[0]
        if len(idx) < 6 or (roads and len(idx) < 0.12 * n_pool):
            break
        best_in, best_score = None, 0
        prob = span[idx] / span[idx].sum()
        for _ in range(150):
            seed = rng.choice(idx, size=min(4, len(idx)), replace=False, p=prob)
            sm = np.zeros(tr.n, bool); sm[seed] = True
            f = fit_geometry(subset(tr, sm), y_h, 2, iters=6)
            if f is None:
                continue
            res, _ = _track_residual(subset(tr, remaining), f[0], f[1], y_h, p0)
            inl = res < inlier_lane
            sc = span[idx][inl].sum()
            if sc > best_score:
                best_score, best_in = sc, idx[inl]
        min_tracks = 5 if not roads else max(8, int(0.15 * n_pool))  # extra roads must be substantial
        if best_in is None or len(best_in) < min_tracks:
            break
        m = np.zeros(tr.n, bool); m[best_in] = True
        for _ in range(3):                                 # refine on consensus set
            cd = choose_degree(subset(tr, m), y_h)
            if cd is None:
                m[:] = False
                break
            deg, f = cd
            res, _ = _track_residual(subset(tr, remaining), f[0], f[1], y_h, p0)
            m = np.zeros(tr.n, bool); m[idx[res < inlier_lane]] = True
            if m.sum() < 5:
                break
        if m.sum() < min_tracks:
            break
        # same flat road as an earlier one (e.g. the opposite carriageway)? then merge
        merged = False
        for j, rm in enumerate(roads):
            if _merge_cost(tr, rm, m, y_h) < 1.35:
                roads[j] = rm | m
                merged = True
                break
        if not merged:
            roads.append(m)
        remaining &= ~m
        if verbose:
            print(f"   road {len(roads)}: {m.sum()} tracks (degree {deg}), {remaining.sum()} left")
    return roads


# --------------------------------------------------------------------------- lanes

@dataclass
class Carriageway:
    direction: str
    boundaries_u: list
    centers_u: list
    lane_width_u: float
    periodicity: float        # 0..1, how strongly traffic forms lanes
    mode: str                 # "traffic-lanes" or "virtual (no lane discipline)"
    support: list             # share of traffic per lane
    weight: float
    n_tracks: int = 0
    confidence: float = 0.0   # 0..1


@dataclass
class Road:
    geometry: Geometry
    carriageways: list
    degree: int
    n_tracks: int

    def all_boundaries(self):
        return [b for cw in self.carriageways for b in cw.boundaries_u]


@dataclass
class LaneModel:
    roads: list
    y_h: float
    car_width_u: float
    lane_width_prior_u: float
    n_tracks: int
    meta: dict = field(default_factory=dict)

    @property
    def n_lanes(self):
        return sum(len(c.centers_u) for r in self.roads for c in r.carriageways)

    def to_json(self, path, W, H):
        out = dict(model="x = g(y) + u*(y - y_h)   (per road; lane lines are constant u)",
                   frame=[W, H], y_h=self.y_h, car_width_u=self.car_width_u,
                   lane_width_prior_u=self.lane_width_prior_u, n_tracks=self.n_tracks,
                   n_lanes=self.n_lanes, meta=self.meta, roads=[])
        for r in self.roads:
            g = r.geometry
            ys = np.linspace(max(g.y_top, g.y_h + 5), H - 1, 40)
            rd = dict(g_coef=g.gcoef, g_powers=g.powers, y_top=g.y_top, fit_residual_px=g.resid_px,
                      degree=r.degree, n_tracks=r.n_tracks, carriageways=[])
            for cw in r.carriageways:
                d = asdict(cw)
                d["boundaries_px"] = [[[round(float(g.x_of(b, y)), 1), round(float(y), 1)] for y in ys]
                                      for b in cw.boundaries_u]
                rd["carriageways"].append(d)
            out["roads"].append(rd)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)


def _wquantile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, cw, v[o]))


def comb_fit(u, w, p0):
    """Best evenly spaced comb (period p, phase phi) for weighted samples u."""
    best = (-1.0, p0, 0.0, 0.0)
    for p in np.linspace(0.65 * p0, 1.5 * p0, 120):
        z = np.sum(w * np.exp(2j * np.pi * u / p)) / w.sum()
        R = abs(z)
        score = R * np.exp(-np.log(p / p0) ** 2 / (2 * 0.3 ** 2))
        if score > best[0]:
            best = (score, p, (np.angle(z) / (2 * np.pi) * p) % p, R)
    _, p, phi, R = best
    return p, phi, R


def split_carriageways(u, w, dirs, p0):
    """Group chunks by direction, then split each direction at empty gaps."""
    groups = []
    for d in (1, -1):
        m = dirs == d
        if w[m].sum() < 0.08 * w.sum():
            continue
        um, wm = u[m], w[m]
        o = np.argsort(um)
        us, ws = um[o], wm[o]
        # gap = long stretch with (almost) no traffic
        cuts = [0]
        for i in range(1, len(us)):
            if us[i] - us[i - 1] > 0.9 * p0:
                cuts.append(i)
        cuts.append(len(us))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if ws[a:b].sum() >= 0.04 * w.sum() and b - a >= 6:
                groups.append((d, np.where(m)[0][o[a:b]]))
    # a minority-direction group sitting inside a bigger one = wrong-way riders, not a carriageway
    keep = []
    for i, (d, idx) in enumerate(groups):
        lo, hi = np.percentile(u[idx], [10, 90])
        inside = False
        for j, (d2, idx2) in enumerate(groups):
            if j == i or w[idx2].sum() <= w[idx].sum():
                continue
            lo2, hi2 = np.percentile(u[idx2], [5, 95])
            if lo >= lo2 - 0.3 * p0 and hi <= hi2 + 0.3 * p0:
                inside = True
        if not inside:
            keep.append((d, idx))
    return keep


def kde(u, w, grid, bw):
    h = np.zeros_like(grid)
    for ui, wi in zip(u, w):
        h += wi * np.exp(-0.5 * ((grid - ui) / bw) ** 2)
    return h


def traffic_peaks(u, w, p0):
    """Lane centres = peaks of the lateral traffic density."""
    from scipy.signal import find_peaks
    lo, hi = np.percentile(u, [0.5, 99.5])
    grid = np.linspace(lo - 0.5 * p0, hi + 0.5 * p0, 800)
    du = grid[1] - grid[0]
    h = kde(u, w, grid, 0.07 * p0)
    pk, pr = find_peaks(h, height=0.12 * h.max(), prominence=0.10 * h.max(), distance=max(1, int(0.55 * p0 / du)))
    # a lane must carry a real share of the traffic, not a handful of odd tracks
    keep = []
    for k in pk:
        share = w[np.abs(u - grid[k]) < 0.3 * p0].sum() / w.sum()
        if share >= 0.08:
            keep.append(k)
    pk = np.array(keep, int)
    return grid[pk], h[pk], grid, h


def lanes_for_group(u, w, p0, car_u, tracks=None):
    """Lane width: the physical prior p0 (= 1.95 car widths) unless the traffic
    peaks agree on a spacing within 20 % of it.  Lane positions: phase of the
    evenly spaced comb that best explains where the traffic actually drives."""
    lo = _wquantile(u, w, 0.02)
    hi = _wquantile(u, w, 0.98)
    peaks, heights, grid, h = traffic_peaks(u, w, p0)
    p = p0
    if len(peaks) >= 2:
        sp = np.diff(peaks)
        p_est = float(np.median(sp / np.maximum(np.round(sp / p0), 1)))
        R_est = abs(np.sum(w * np.exp(2j * np.pi * u / p_est))) / w.sum()
        # clear, regularly spaced traffic streams beat the car-size prior
        if 0.55 <= p_est / p0 <= 1.5 and (R_est >= 0.35 or abs(p_est / p0 - 1) < 0.2):
            p = p_est
    z = np.sum(w * np.exp(2j * np.pi * u / p)) / w.sum()
    R = float(abs(z))                                      # 1 = traffic sits exactly on lane centres
    if len(peaks) >= 1 and R >= 0.25:
        phi = (np.angle(z) / (2 * np.pi) * p) % p
        k0, k1 = int(np.floor((lo - phi) / p)) - 1, int(np.ceil((hi - phi) / p)) + 1
        centers = [phi + k * p for k in range(k0, k1 + 1) if lo - 0.45 * p <= phi + k * p <= hi + 0.45 * p]
        mode = "traffic-lanes"
    else:                                                  # no lane discipline: fill the used width
        if tracks is not None:                             # road extent from per-vehicle positions
            ids = np.unique(tracks)
            ut = np.array([np.median(u[tracks == k]) for k in ids])
            lo, hi = np.percentile(ut, [4, 96])
        width = hi - lo + car_u
        n = max(1, int(round(width / p0)))
        p = width / n
        centers = [lo - car_u / 2 + p * (k + 0.5) for k in range(n)]
        mode = "virtual (no lane discipline)"
    centers = np.array(sorted(centers))
    bounds = [centers[0] - p / 2] + list((centers[1:] + centers[:-1]) / 2) + [centers[-1] + p / 2]
    lane_of = np.clip(np.searchsorted(np.array(bounds), u) - 1, 0, len(centers) - 1)
    support = np.bincount(lane_of, w, len(centers)) / w.sum()
    while len(centers) > 1 and support[0] < 0.04:          # comb teeth outside the used road
        centers, support, bounds = centers[1:], support[1:], bounds[1:]
    while len(centers) > 1 and support[-1] < 0.04:
        centers, support, bounds = centers[:-1], support[:-1], bounds[:-1]
    return dict(boundaries_u=[float(b) for b in bounds], centers_u=[float(c) for c in centers],
                lane_width_u=float(p), periodicity=R, mode=mode,
                support=[round(float(s), 3) for s in support])


def refine_horizon(ch: Chunks, y_h0: float, H: int):
    """On a straight road the tracks converge to a vanishing point, which pins
    y_h better than box widths do.  Only trust it when a straight model fits."""
    y_min = float(np.percentile(ch.y, 1))
    span = H - y_min
    cands = np.linspace(max(y_h0 - 0.5 * span, y_min - 3 * span), min(y_h0 + 0.5 * span, y_min - 0.005 * span), 41)
    res = [fit_geometry(ch, v, 1, iters=12) for v in cands]
    res = np.array([np.inf if f is None else f[3] for f in res])
    k = int(np.argmin(res))
    if 0 < k < len(cands) - 1:                             # interior minimum = real VP
        return float(cands[k])
    return y_h0


def widen_virtual(cws, geo, x1, y2, x2, H, p0, car_u):
    """Without lane discipline the lanes are just the used road width split into
    standard lanes, so the width must include vehicles that queue or stand
    still (e.g. a whole lane waiting at a red light), not only moving tracks."""
    ok = (y2 > geo.y_h + 3) & (y2 < H - 3)
    ua = ((x1 + x2) / 2 - geo.g(y2))[ok] / (y2[ok] - geo.y_h)
    for i, c in enumerate(cws):
        if not c.mode.startswith("virtual"):
            continue
        lo, hi = c.boundaries_u[0], c.boundaries_u[-1]
        lim_lo = lo - p0 if i == 0 else max(lo - p0, (cws[i - 1].boundaries_u[-1] + lo) / 2)
        lim_hi = hi + p0 if i == len(cws) - 1 else min(hi + p0, (hi + cws[i + 1].boundaries_u[0]) / 2)
        near = ua[(ua > lim_lo) & (ua < lim_hi)]
        if len(near) < 50:
            continue
        lo2 = min(lo, float(np.percentile(near, 2)) - car_u / 2)
        hi2 = max(hi, float(np.percentile(near, 98)) + car_u / 2)
        n = max(1, int(round((hi2 - lo2) / p0)))
        p = (hi2 - lo2) / n
        c.boundaries_u = [float(lo2 + k * p) for k in range(n + 1)]
        c.centers_u = [float(lo2 + (k + 0.5) * p) for k in range(n)]
        c.lane_width_u = float(p)


def car_width_u(det, W, H, y_h):
    """Median car width in u-units.  When a road runs diagonally across the
    image (track slope s = dx/dy), the box width also contains part of the car's
    length; the empirical correction w / (1 + 0.19 max(0, |s| - 0.5)) removes
    most of it and leaves the usual 'road goes up the image' view (|s| < 0.5)
    untouched."""
    tid, cls, x1, y2, x2 = det[:, 1], det[:, 2], det[:, 3], det[:, 6], det[:, 5]
    m = (cls == CAR) & (x1 > 2) & (x2 < W - 3) & (y2 < H - 3) & (y2 > y_h + 5)
    if m.sum() < 10:
        m = np.isin(cls, [CAR, BUS, TRUCK]) & (x1 > 2) & (x2 < W - 3) & (y2 > y_h + 5)
    slope = np.zeros(len(det))
    for k in np.unique(tid[m]):
        t = m & (tid == k)
        if t.sum() >= 4 and np.ptp(y2[t]) > 10:
            slope[t] = np.polyfit(y2[t], (x1[t] + x2[t]) / 2, 1)[0]
    w = (x2 - x1)[m] / (1 + 0.19 * np.maximum(np.abs(slope[m]) - 0.5, 0))
    return float(np.median(w / (y2[m] - y_h)))


def infer_lanes(det: np.ndarray, W: int, H: int, verbose=False, seed=0) -> LaneModel:
    rng = np.random.default_rng(seed)
    ch = build_chunks(det, W, H)                           # pass 1: box bottom-centres
    y_min = float(np.percentile(ch.y, 1))
    y_h = horizon_from_widths(det, W, H)
    if y_h is None or y_h > y_min - 0.005 * (H - y_min):   # vehicles may be seen almost at the horizon
        y_h = y_min - 0.8 * (H - y_min)
    y_h_w = y_h
    y_h = refine_horizon(ch, y_h, H)
    car_u = car_width_u(det, W, H, y_h)
    # pass 2: proper ground-contact points, then re-estimate the horizon
    ch = build_chunks(det, W, H, y_h=y_h, car_u=car_u)
    y_h = refine_horizon(ch, y_h, H)
    car_u = car_width_u(det, W, H, y_h)
    ch = build_chunks(det, W, H, y_h=y_h, car_u=car_u)
    tr = build_chunks(det, W, H, chunk_s=1e9, y_h=y_h, car_u=car_u)   # one chunk per track
    p0 = LANE_PER_CAR * car_u
    x1, y2, x2 = det[:, 3], det[:, 6], det[:, 5]
    if verbose:
        print(f"   y_h: width-based {y_h_w:.0f} -> refined {y_h:.0f};  car_u={car_u:.4f}  p0={p0:.4f}")
    road_masks = find_roads_by_direction(tr, y_h, p0, rng, verbose=verbose)
    roads = []
    for rm in road_masks:
        tids = set(tr.ctrack[rm].tolist())
        rch = subset(ch, np.array([t in tids for t in ch.ctrack]))
        # road shape from whole tracks (1 s chunks of crawling traffic carry no direction)
        cd = choose_degree(subset(tr, rm), y_h, verbose)
        if cd is None:
            continue
        deg, (coef, powers, _, resid) = cd
        # lateral position of every 1 s chunk given that shape (lane changes stay separate)
        dch = np.maximum(rch.y - y_h, 1)
        gx = sum(c * dch ** p for c, p in zip(coef, powers))
        u = np.bincount(rch.cid, (rch.x - gx) * dch, rch.n) / (np.bincount(rch.cid, dch * dch, rch.n) + 1e-9)
        geo = Geometry(y_h, [float(c) for c in coef], powers, float(np.percentile(rch.y, 2)), float(H), resid)
        cws = []
        for d, idx in split_carriageways(u, rch.cw, rch.cdir, p0):
            L = lanes_for_group(u[idx], rch.cw[idx], p0, car_u, rch.ctrack[idx])
            nt = len(np.unique(rch.ctrack[idx]))
            conf = min(1.0, nt / 40) * (0.35 + 0.65 * min(1.0, L["periodicity"] / 0.45))
            if L["mode"].startswith("virtual"):
                conf = min(conf, 0.35)
            cws.append(Carriageway(direction="towards camera" if d > 0 else "away from camera",
                                   weight=float(rch.cw[idx].sum()), n_tracks=nt, confidence=round(conf, 2), **L))
        cws.sort(key=lambda c: c.boundaries_u[0])
        if not cws:
            continue
        widen_virtual(cws, geo, x1, y2, x2, H, p0, car_u)
        # draw as far as the road is seen: every detection (also stopped / queued
        # vehicles) that lies inside this road's lateral range extends it
        ub = [b for c in cws for b in c.boundaries_u]
        ok = (y2 > y_h + 3) & (y2 < H - 3)
        cx = (x1 + x2) / 2
        ua = (cx[ok] - geo.g(y2[ok])) / (y2[ok] - y_h)
        inside = (ua > min(ub)) & (ua < max(ub))
        if inside.sum() > 20:
            far = float(np.percentile(y2[ok][inside], 0.5))
            if deg > 1:                                    # do not extrapolate a curve far
                far = max(far, geo.y_top - 0.15 * (H - geo.y_top))
            geo.y_top = min(geo.y_top, far)
        roads.append(Road(geo, cws, deg, len(tids)))
        if verbose:
            print(f"   road {len(roads)}: degree {deg}, resid {resid:.1f}px, {len(tids)} tracks")
            for c in cws:
                print(f"      {c.direction:17s} {len(c.centers_u)} lanes  p={c.lane_width_u:.4f} "
                      f"R={c.periodicity:.2f} conf={c.confidence} {c.mode} support={c.support}")
    return LaneModel(roads, y_h, car_u, p0, int(tr.n), meta=dict(y_h_width=y_h_w))
