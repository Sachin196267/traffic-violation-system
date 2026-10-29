import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Sky } from 'three/examples/jsm/objects/Sky.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

/*
 Single-camera 3D reconstruction of an analysed video.
 A pixel (u, v) below the horizon y_h lies on the (flat) road at
     depth Z = h * F / (v - y_h),   lateral X = (u - cx) * h / (v - y_h)
 The camera height h is estimated so that tracked cars come out ~1.8 m wide; F ~ image width.
 Layout, lanes, positions and motion come from the real tracks; absolute scale is approximate.
*/
export type TwinData = { meta: any; tracks: { id: number; type: string; path: number[][]; plate: string | null }[]; findings: any[] }
type Opt = { labels: boolean; onlyViol: boolean }

const DIM: Record<string, [number, number, number]> = {   // width, length, height (m)
  car: [1.75, 4.2, 1.5], motorcycle: [0.75, 2.0, 1.15], bicycle: [0.6, 1.75, 1.05], autorickshaw: [1.35, 2.7, 1.8],
  bus: [2.5, 11, 3.1], truck: [2.45, 8, 3.2], caravan: [2.1, 5.6, 2.5], trailer: [2.45, 10, 3.3], 'vehicle fallback': [1.75, 4.2, 1.5],
}
const PAINT = [0xf5f6f7, 0xf5f6f7, 0xc9ccd1, 0xa7adb5, 0x2b2f36, 0x8d1c1c, 0x1f4e8c, 0x6d7278, 0xe8e2d4]
const ROUTE_COL: Record<string, number> = { confirmed: 0xef4444, review: 0xf59e0b, exempt: 0x3b82f6 }
const rnd = (seed: number) => () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646 }

function noiseTexture(base: [number, number, number], spread: number, size = 256, rep = 20) {
  const c = document.createElement('canvas'); c.width = c.height = size
  const g = c.getContext('2d')!, im = g.createImageData(size, size), r = rnd(7)
  for (let i = 0; i < size * size; i++) {
    const n = (r() - 0.5) * spread
    im.data[i * 4] = base[0] + n; im.data[i * 4 + 1] = base[1] + n; im.data[i * 4 + 2] = base[2] + n; im.data[i * 4 + 3] = 255
  }
  g.putImageData(im, 0, 0)
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rep, rep); t.colorSpace = THREE.SRGBColorSpace
  return t
}

const M = {
  glass: new THREE.MeshPhysicalMaterial({ color: 0x1b2430, roughness: 0.05, metalness: 0.6, clearcoat: 1 }),
  tyre: new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.85 }),
  rim: new THREE.MeshStandardMaterial({ color: 0xb8bec6, roughness: 0.3, metalness: 0.9 }),
  head: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff4d6, emissiveIntensity: 1.2 }),
  tail: new THREE.MeshStandardMaterial({ color: 0xb91c1c, emissive: 0xff2020, emissiveIntensity: 0.8 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x202328, roughness: 0.7 }),
  skin: new THREE.MeshStandardMaterial({ color: 0x8d5a3b, roughness: 0.8 }),
}
const paint = (c: number) => new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.35, metalness: 0.4, clearcoat: 0.8, clearcoatRoughness: 0.15 })
const box = (w: number, h: number, l: number, m: THREE.Material, x = 0, y = 0, z = 0, r = 0.08) => {
  const g = new THREE.Mesh(roundedBox(w, h, l, r), m); g.position.set(x, y, z); g.castShadow = true; g.receiveShadow = true; return g
}
function roundedBox(w: number, h: number, l: number, r: number) {
  // extruded rounded rectangle (w x h) along z (length l)
  const s = new THREE.Shape(), x = -w / 2, y = -h / 2, rr = Math.min(r, w / 3, h / 3)
  s.moveTo(x + rr, y); s.lineTo(x + w - rr, y); s.quadraticCurveTo(x + w, y, x + w, y + rr); s.lineTo(x + w, y + h - rr)
  s.quadraticCurveTo(x + w, y + h, x + w - rr, y + h); s.lineTo(x + rr, y + h); s.quadraticCurveTo(x, y + h, x, y + h - rr)
  s.lineTo(x, y + rr); s.quadraticCurveTo(x, y, x + rr, y)
  const g = new THREE.ExtrudeGeometry(s, { depth: l - 2 * rr, bevelEnabled: true, bevelSize: rr * 0.6, bevelThickness: rr, bevelSegments: 3, curveSegments: 4 })
  g.translate(0, 0, -(l - 2 * rr) / 2); return g
}
function wheel(rad: number, wid: number, x: number, z: number) {
  const g = new THREE.Group()
  const t = new THREE.Mesh(new THREE.CylinderGeometry(rad, rad, wid, 20), M.tyre); t.rotation.z = Math.PI / 2; t.castShadow = true; g.add(t)
  const r = new THREE.Mesh(new THREE.CylinderGeometry(rad * 0.6, rad * 0.6, wid * 1.02, 14), M.rim); r.rotation.z = Math.PI / 2; g.add(r)
  g.position.set(x, rad, z); return g
}

function vehicle(type: string, seed: number): THREE.Group {
  const [w, l, h] = DIM[type] || DIM.car, r = rnd(seed * 31 + 11), g = new THREE.Group()
  if (type === 'motorcycle' || type === 'bicycle') {
    const body = paint([0x1f2937, 0x991b1b, 0x1e3a8a, 0x111827, 0x6b7280][Math.floor(r() * 5)])
    g.add(wheel(0.32, 0.12, 0, -l / 2 + 0.35), wheel(0.32, 0.12, 0, l / 2 - 0.35))
    g.add(box(0.28, 0.32, l * 0.62, body, 0, 0.62, 0, 0.1))                   // tank + seat
    g.add(box(0.05, 0.5, 0.05, M.rim, 0, 0.75, l / 2 - 0.45, 0.02))            // fork
    g.add(box(0.7, 0.04, 0.04, M.dark, 0, 1.02, l / 2 - 0.5, 0.01))            // handlebar
    const light = box(0.16, 0.12, 0.06, M.head, 0, 0.9, l / 2 - 0.36, 0.03); g.add(light)
    const shirt = new THREE.MeshStandardMaterial({ color: [0x2563eb, 0xf8fafc, 0x16a34a, 0xdc2626, 0x78350f, 0x334155][Math.floor(r() * 6)], roughness: 0.9 })
    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.42, 4, 10), shirt); torso.position.set(0, 1.2, -0.05); torso.rotation.x = 0.25; torso.castShadow = true; g.add(torso)
    const headM = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 12), M.skin); headM.position.set(0, 1.62, 0.05); headM.castShadow = true; g.add(headM)
    for (const s of [-1, 1]) { const leg = new THREE.Mesh(new THREE.CapsuleGeometry(0.08, 0.4, 4, 8), M.dark); leg.position.set(s * 0.17, 0.72, 0.12); leg.rotation.x = 1.1; g.add(leg) }
    return g
  }
  if (type === 'autorickshaw') {
    const yellow = paint(0xf2c200), green = paint(r() < 0.5 ? 0x1f7a3a : 0x1b1b1b)
    g.add(wheel(0.25, 0.15, 0, l / 2 - 0.35), wheel(0.25, 0.15, -w / 2 + 0.12, -l / 2 + 0.55), wheel(0.25, 0.15, w / 2 - 0.12, -l / 2 + 0.55))
    g.add(box(w, 0.55, l * 0.9, green, 0, 0.55, -0.05, 0.12))
    g.add(box(w * 0.96, 0.08, l * 0.95, yellow, 0, h - 0.1, -0.05, 0.1))        // canopy roof
    g.add(box(w * 0.96, 0.9, 0.06, yellow, 0, h - 0.55, -l / 2 + 0.1, 0.03))    // back panel
    g.add(box(w * 0.9, 0.6, 0.05, M.glass, 0, h - 0.5, l / 2 - 0.55, 0.02))     // windscreen
    g.add(box(0.2, 0.12, 0.05, M.head, 0, 0.8, l / 2 - 0.1, 0.03))
    return g
  }
  const big = type === 'bus' || type === 'truck' || type === 'trailer' || type === 'caravan'
  const col = type === 'bus' ? [0xb91c1c, 0x1d4ed8, 0xf5f6f7, 0x15803d][Math.floor(r() * 4)] : type === 'truck' || type === 'trailer' ? [0xea580c, 0xca8a04, 0x1d4ed8, 0xb91c1c][Math.floor(r() * 4)] : PAINT[Math.floor(r() * PAINT.length)]
  const body = paint(col), wr = big ? 0.5 : 0.33
  for (const z of big ? [l / 2 - 1.3, -l / 2 + 1.4, -l / 2 + 2.6] : [l / 2 - 0.8, -l / 2 + 0.8])
    for (const s of [-1, 1]) g.add(wheel(wr, 0.26, s * (w / 2 - 0.12), z))
  if (type === 'bus') {
    g.add(box(w, h - 0.35, l, body, 0, 0.35 + (h - 0.35) / 2, 0, 0.2))
    for (const s of [-1, 1]) g.add(box(0.02, 0.85, l * 0.82, M.glass, s * (w / 2 + 0.005), h * 0.62, -0.2, 0.02))
    g.add(box(w * 0.92, 1.0, 0.04, M.glass, 0, h * 0.62, l / 2 + 0.005, 0.02))
  } else if (type === 'truck' || type === 'trailer') {
    const cabL = 2.2
    g.add(box(w, 2.1, cabL, body, 0, 0.45 + 1.05, l / 2 - cabL / 2, 0.15))
    g.add(box(w * 0.9, 0.8, 0.04, M.glass, 0, 1.95, l / 2 + 0.005, 0.02))
    g.add(box(w, h - 0.5, l - cabL - 0.2, paint(r() < 0.5 ? 0xd6d3c8 : 0x3f6212), 0, 0.5 + (h - 0.5) / 2, -cabL / 2 - 0.1, 0.06))
  } else {
    const low = h * 0.4, base = 0.3
    g.add(box(w, low, l, body, 0, base + low / 2, 0, 0.2))                                   // lower body
    const cabL = l * 0.44, cabH = h * 0.36, cabZ = -l * 0.06, cabY = base + low + cabH / 2 - 0.04
    g.add(box(w * 0.84, cabH, cabL, body, 0, cabY, cabZ, 0.22))                             // roof / pillars
    for (const s of [-1, 1]) g.add(box(0.02, cabH * 0.62, cabL * 0.84, M.glass, s * (w * 0.42 + 0.012), cabY + 0.02, cabZ, 0.03))   // side windows
    const ws = box(w * 0.76, cabH * 0.75, 0.03, M.glass, 0, cabY, cabZ + cabL / 2 + 0.09, 0.03); ws.rotation.x = -0.55; g.add(ws)   // windscreen
    const rs = box(w * 0.74, cabH * 0.7, 0.03, M.glass, 0, cabY, cabZ - cabL / 2 - 0.08, 0.03); rs.rotation.x = 0.5; g.add(rs)
    for (const z of [l / 2 + 0.02, -l / 2 - 0.02]) g.add(box(w * 0.96, 0.16, 0.1, M.dark, 0, base + 0.12, z, 0.04))                 // bumpers
    g.add(box(w * 0.46, 0.14, 0.04, M.dark, 0, base + low * 0.62, l / 2 + 0.01, 0.02))                                                // grille
    const plate = new THREE.MeshStandardMaterial({ color: 0xfafafa, roughness: 0.4 })
    for (const z of [l / 2 + 0.08, -l / 2 - 0.08]) g.add(box(0.5, 0.12, 0.02, plate, 0, base + 0.2, z, 0.01))
  }
  for (const s of [-1, 1]) {
    g.add(box(0.28, 0.12, 0.05, M.head, s * (w / 2 - 0.3), big ? 0.75 : 0.62, l / 2 + 0.01, 0.03))
    g.add(box(0.26, 0.12, 0.05, M.tail, s * (w / 2 - 0.3), big ? 0.9 : 0.68, -l / 2 - 0.01, 0.03))
  }
  return g
}

function spriteLabel(text: string, bg: string) {
  const c = document.createElement('canvas'), x = c.getContext('2d')!
  x.font = '600 30px Inter, sans-serif'; const w = Math.ceil(x.measureText(text).width) + 28
  c.width = w; c.height = 48; x.fillStyle = bg; x.beginPath(); x.roundRect(0, 0, w, 48, 12); x.fill()
  x.fillStyle = '#fff'; x.font = '600 30px Inter, sans-serif'; x.fillText(text, 14, 33)
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, depthTest: false })); s.scale.set(w / 48 * 0.8, 0.8, 1); s.renderOrder = 20
  return s
}

// offset a ground polyline sideways by d metres (d > 0 = to the right when walking along it)
function offsetLine(p: THREE.Vector3[], d: number) {
  return p.map((q, i) => {
    const a = p[Math.max(0, i - 1)], b = p[Math.min(p.length - 1, i + 1)]
    const dir = b.clone().sub(a).setY(0).normalize(), n = new THREE.Vector3(-dir.z, 0, dir.x)
    return q.clone().addScaledVector(n, d)
  })
}
function strip(L: THREE.Vector3[], R: THREE.Vector3[], y: number, mat: THREE.Material, uvScale = 0.25) {
  const pos: number[] = [], uv: number[] = []
  for (let i = 0; i < L.length - 1; i++) {
    const q = [L[i], L[i + 1], R[i], R[i + 1]]
    for (const k of [0, 1, 2, 1, 3, 2]) { pos.push(q[k].x, y, q[k].z); uv.push(q[k].x * uvScale, q[k].z * uvScale) }
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2)); g.computeVertexNormals()
  const m = new THREE.Mesh(g, mat); m.receiveShadow = true; return m
}
function dashes(line: THREE.Vector3[], dash: number, gap: number, width: number, mat: THREE.Material, g: THREE.Group) {
  let acc = 0, on = true
  for (let i = 0; i < line.length - 1; i++) {
    const a = line[i], b = line[i + 1], seg = a.distanceTo(b)
    let s = 0
    while (s < seg) {
      const len = Math.min((on ? dash : gap) - acc, seg - s)
      if (on && len > 0.05) {
        const p1 = a.clone().lerp(b, s / seg), p2 = a.clone().lerp(b, (s + len) / seg), mid = p1.clone().add(p2).multiplyScalar(0.5)
        const m = new THREE.Mesh(new THREE.PlaneGeometry(width, p1.distanceTo(p2)), mat)
        m.rotation.x = -Math.PI / 2; m.rotation.z = Math.atan2(p2.x - p1.x, p2.z - p1.z); m.position.set(mid.x, 0.025, mid.z); m.receiveShadow = true; g.add(m)
      }
      acc += len; s += len
      if (acc >= (on ? dash : gap) - 1e-6) { acc = 0; on = !on }
    }
  }
}

export default function Twin3D({ data, lanes, t, opt, view }: { data: TwinData; lanes: any; t: number; opt: Opt; view: string }) {
  const host = useRef<HTMLDivElement>(null)
  const st = useRef<any>({})
  const [gen, setGen] = useState(0)          // bumps when the scene is rebuilt, so vehicles get placed again
  const W = data.meta?.width || 1280, Hh = data.meta?.height || 720, cx = W / 2
  const yh = lanes?.y_h ?? Hh * 0.28

  useEffect(() => {
    const el = host.current!
    // camera height from car widths: w_px * h / (v - y_h) should be ~1.75 m
    const hs: number[] = []
    for (const tr of data.tracks) if (tr.type === 'car') for (const q of tr.path) { const dv = q[4] - yh; const wp = q[3] - q[1]; if (dv > 10 && wp > 8) hs.push(1.75 * dv / wp) }
    hs.sort((a, b) => a - b)
    const hcam = hs.length > 10 ? Math.min(20, Math.max(2.5, hs[Math.floor(hs.length / 2)])) : 6
    const F = W * 0.9
    const ground = (u: number, v: number) => { const dv = Math.max(v - yh, Hh * 0.02); return new THREE.Vector3((u - cx) * hcam / dv, 0, -Math.min(160, hcam * F / dv)) }

    const w = el.clientWidth, h = Math.round(w * 0.58)
    const r = new THREE.WebGLRenderer({ antialias: true }); r.setPixelRatio(Math.min(2, devicePixelRatio)); r.setSize(w, h)
    r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap; r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 0.9
    el.appendChild(r.domElement)
    const scene = new THREE.Scene()
    const pm = new THREE.PMREMGenerator(r); scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture
    const sky = new Sky(); sky.scale.setScalar(4000); scene.add(sky)
    const u = sky.material.uniforms; u.turbidity.value = 4; u.rayleigh.value = 1.2; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.8
    const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(55), THREE.MathUtils.degToRad(140)); u.sunPosition.value.copy(sunDir)
    scene.fog = new THREE.Fog(0xcfdcea, 120, 320)
    scene.add(new THREE.HemisphereLight(0xdfeaff, 0x6b7d5a, 0.9))
    const sun = new THREE.DirectionalLight(0xfff3e0, 2.4); sun.castShadow = true; sun.shadow.mapSize.set(4096, 4096); sun.shadow.bias = -0.0004
    Object.assign(sun.shadow.camera, { left: -90, right: 90, top: 90, bottom: -90, far: 400 }); scene.add(sun); scene.add(sun.target)

    const grass = new THREE.Mesh(new THREE.PlaneGeometry(800, 800), new THREE.MeshStandardMaterial({ map: noiseTexture([92, 128, 70], 34, 256, 80), roughness: 1 }))
    grass.rotation.x = -Math.PI / 2; grass.receiveShadow = true; scene.add(grass)
    const asphalt = new THREE.MeshStandardMaterial({ map: noiseTexture([72, 75, 80], 22), roughness: 0.92 })
    const paving = new THREE.MeshStandardMaterial({ map: noiseTexture([182, 176, 164], 26), roughness: 0.95 })
    const kerbM = new THREE.MeshStandardMaterial({ color: 0xd9d9d6, roughness: 0.8 })
    const white = new THREE.MeshStandardMaterial({ color: 0xf4f4f0, roughness: 0.6, emissive: 0x222222 })
    const env = new THREE.Group(); scene.add(env)
    const bb = new THREE.Box3()
    const cws = (lanes?.roads || []).flatMap((rd: any) => rd.carriageways || [])
    const edges: THREE.Vector3[][] = []
    for (const cw of cws) {
      const extend = (pts: THREE.Vector3[]) => {
        if (pts.length < 2) return pts
        const d0 = pts[0].clone().sub(pts[1]).setY(0).normalize(), d1 = pts[pts.length - 1].clone().sub(pts[pts.length - 2]).setY(0).normalize()
        const far = [3, 2, 1].map(k => pts[0].clone().addScaledVector(d0, k * 8)), near = [1, 2, 3].map(k => pts[pts.length - 1].clone().addScaledVector(d1, k * 5))
        return [...far, ...pts, ...near]
      }
      const B: THREE.Vector3[][] = (cw.boundaries_px || []).map((line: number[][]) => extend(line.map(p => ground(p[0], p[1]))).filter(p => p.z > -170))
      if (B.length < 2 || B[0].length < 2) continue
      const L = B[0], R = B[B.length - 1]
      if (L.length !== R.length) continue
      env.add(strip(L, R, 0.01, asphalt))
      L.concat(R).forEach(p => bb.expandByPoint(p))
      B.forEach((line, i) => {
        if (i === 0 || i === B.length - 1) dashes(line, 1e9, 0, 0.18, white, env)      // solid edge lines
        else dashes(line, 3, 4.5, 0.14, white, env)                                     // IRC lane dashes: 3 m line, 4.5 m gap
      })
      edges.push(L, R)
    }
    let apronEdges: THREE.Vector3[][] | null = null
    {   // asphalt under everywhere vehicles actually drove (lanes the lane model did not learn, e.g. the far carriageway)
      const pts: THREE.Vector3[] = []
      for (const tr of data.tracks) for (const q of tr.path.filter((_, i) => i % 3 === 0)) { const g0 = ground((q[1] + q[3]) / 2, q[4]); if (g0.z > -150) pts.push(g0) }
      if (pts.length > 10) {
        let ang = 0
        if (edges.length) { const e = edges[0]; ang = Math.atan2(e[e.length - 1].x - e[0].x, e[e.length - 1].z - e[0].z) }
        else { const a0 = pts[0], a1 = pts[pts.length - 1]; ang = Math.atan2(a1.x - a0.x, a1.z - a0.z) }
        const ca = Math.cos(ang), sa = Math.sin(ang)
        const us = pts.map(p => p.x * sa + p.z * ca).sort((a, b) => a - b), vs = pts.map(p => p.x * ca - p.z * sa).sort((a, b) => a - b)
        const q = (arr: number[], f: number) => arr[Math.floor(f * (arr.length - 1))]
        const u0 = q(us, 0.02) - 20, u1 = q(us, 0.98) + 20, v0 = q(vs, 0.03) - 2.5, v1 = q(vs, 0.97) + 2.5
        const um = (u0 + u1) / 2, vm = (v0 + v1) / 2
        const apron = new THREE.Mesh(new THREE.PlaneGeometry(v1 - v0, u1 - u0), asphalt)
        apron.rotation.x = -Math.PI / 2; apron.rotation.z = ang
        apron.position.set(um * sa + vm * ca, 0.005, um * ca - vm * sa); apron.receiveShadow = true; env.add(apron)
        const line = (v: number) => { const out: THREE.Vector3[] = []; for (let u = u0; u <= u1; u += 4) out.push(new THREE.Vector3(u * sa + v * ca, 0, u * ca - v * sa)); return out }
        apronEdges = [line(v1), line(v0)]      // road sides: kerbs, footpaths and trees go outside these
        pts.forEach(p => bb.expandByPoint(p))
      }
    }
    if (!edges.length) {        // no lane model: plain road under the traffic
      for (const tr of data.tracks) for (const q of tr.path.filter((_, i) => i % 4 === 0)) bb.expandByPoint(ground((q[1] + q[3]) / 2, q[4]))
      if (!bb.isEmpty()) { const c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3())
        const road = new THREE.Mesh(new THREE.PlaneGeometry(s.x + 4, s.z + 10), asphalt); road.rotation.x = -Math.PI / 2; road.position.set(c.x, 0.01, c.z); road.receiveShadow = true; env.add(road) }
    }
    // kerbs, footpaths, trees and street lights along the outer road edges
    const trunkM = new THREE.MeshStandardMaterial({ color: 0x5b4636, roughness: 1 })
    const leafM = [0x3f7d3a, 0x4b8b3b, 0x356f33].map(c => new THREE.MeshStandardMaterial({ color: c, roughness: 0.9, flatShading: true }))
    const poleM = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.4, metalness: 0.8 })
    const rr = rnd(3)
    const outer = apronEdges || (edges.length ? [edges[0], edges[edges.length - 1]] : [])
    const c0z = bb.isEmpty() ? 0 : bb.max.z
    outer.forEach((edge, k) => {
      const side = k === 0 ? -1 : 1
      const kIn = edge, kOut = offsetLine(edge, side * 0.35), fOut = offsetLine(edge, side * 3.2)
      const kerb = strip(kIn, kOut, 0.15, kerbM); env.add(kerb)
      env.add(strip(kOut, fOut, 0.16, paving))
      let run = 0
      for (let i = 1; i < edge.length; i++) {
        run += edge[i].distanceTo(edge[i - 1])
        if (run < 11) continue
        run = 0
        const p = offsetLine(edge, side * (5.2 + rr() * 2.5))[i]
        if (p.z < -140 || p.z > c0z - 6) continue
        const tree = new THREE.Group(), th = 1.6 + rr() * 1.2
        const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.2, th, 8), trunkM); trunk.position.y = th / 2; trunk.castShadow = true; tree.add(trunk)
        for (let j = 0; j < 3; j++) { const f = new THREE.Mesh(new THREE.IcosahedronGeometry(1.3 + rr() * 0.8, 1), leafM[j % 3]); f.position.set((rr() - 0.5) * 1.2, th + 0.9 + j * 0.5, (rr() - 0.5) * 1.2); f.castShadow = true; tree.add(f) }
        tree.position.copy(p); env.add(tree)
        if (i % 3 === 0) {
          const q = offsetLine(edge, side * 1.6)[i], pole = new THREE.Group()
          const s1 = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, 7, 8), poleM); s1.position.y = 3.5; s1.castShadow = true; pole.add(s1)
          const arm = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 1.6), poleM); arm.position.set(0, 6.9, 0); pole.add(arm)
          pole.position.copy(q); pole.lookAt(q.clone().addScaledVector(offsetLine(edge, -side)[i].sub(q), 1)); env.add(pole)
        }
      }
    })
    for (const tr of data.tracks) for (const q of tr.path.filter((_, i) => i % 6 === 0)) bb.expandByPoint(ground((q[1] + q[3]) / 2, q[4]))
    const c = bb.isEmpty() ? new THREE.Vector3(0, 0, -30) : bb.getCenter(new THREE.Vector3())
    const span = bb.isEmpty() ? 50 : Math.min(140, Math.max(14, bb.getSize(new THREE.Vector3()).length() * 0.55))
    sun.position.copy(c).addScaledVector(sunDir, 150); sun.target.position.copy(c)

    const cam = new THREE.PerspectiveCamera(42, w / h, 0.3, 2000)
    const ctl = new OrbitControls(cam, r.domElement); ctl.enableDamping = true; ctl.maxPolarAngle = Math.PI * 0.48; ctl.minDistance = 4; ctl.maxDistance = 400
    const views: Record<string, () => void> = {
      camera: () => { cam.position.set(0, hcam, 2); ctl.target.set(c.x * 0.3, 0, c.z) },
      bird: () => { cam.position.set(c.x, span * 1.25, c.z + 0.01); ctl.target.copy(c) },
      orbit: () => { cam.position.set(c.x + span * 0.55, span * 0.5, c.z + span * 0.7); ctl.target.copy(c) },
    }
    views.orbit()

    const vehicles = new Map<number, THREE.Group>()
    const rings = new Map<number, THREE.Mesh>()
    const labels = new Map<number, { s: THREE.Sprite; text: string }>()
    for (const tr of data.tracks) {
      const g = vehicle(tr.type, tr.id); g.visible = false; scene.add(g); vehicles.set(tr.id, g)
      const [vw, vl] = DIM[tr.type] || DIM.car
      const ring = new THREE.Mesh(new THREE.RingGeometry(Math.max(vw, vl) * 0.62, Math.max(vw, vl) * 0.62 + 0.35, 40), new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.85, side: THREE.DoubleSide }))
      ring.rotation.x = -Math.PI / 2; ring.visible = false; scene.add(ring); rings.set(tr.id, ring)
    }
    let raf = 0
    const clock = new THREE.Clock()
    const loop = () => {
      ctl.update()
      const k = 0.55 + 0.35 * Math.sin(clock.getElapsedTime() * 5)
      rings.forEach(m => { if (m.visible) (m.material as THREE.MeshBasicMaterial).opacity = k })
      r.render(scene, cam); raf = requestAnimationFrame(loop)
    }
    loop()
    const onResize = () => { const w2 = el.clientWidth, h2 = Math.round(w2 * 0.58); r.setSize(w2, h2); cam.aspect = w2 / h2; cam.updateProjectionMatrix() }
    addEventListener('resize', onResize)
    st.current = { scene, vehicles, rings, labels, ground, views, heading: new Map<number, number>(), cam, bb, hcam, ctl }
    ;(window as any).__twin = st.current
    setGen(x => x + 1)
    return () => { cancelAnimationFrame(raf); removeEventListener('resize', onResize); ctl.dispose(); r.dispose(); pm.dispose(); el.removeChild(r.domElement) }
  }, [data, lanes])

  useEffect(() => { st.current.views?.[view]?.() }, [view, gen])

  useEffect(() => {
    const { scene, vehicles, rings, labels, ground, heading } = st.current; if (!scene) return
    const active = data.findings.filter(f => f.box && Math.abs(f.t - t) < 1.5)
    for (const tr of data.tracks) {
      const m = vehicles.get(tr.id)!, ring = rings.get(tr.id)!, p = tr.path
      const hide = () => { m.visible = false; ring.visible = false; const L = labels.get(tr.id); if (L) L.s.visible = false }
      if (!p.length || t < p[0][0] - 0.1 || t > p[p.length - 1][0] + 0.1) { hide(); continue }
      let i = p.findIndex((q: number[]) => q[0] >= t); if (i < 0) i = p.length - 1
      const a = p[Math.max(0, i - 1)], b = p[i], k = i ? Math.min(1, Math.max(0, (t - a[0]) / Math.max(1e-3, b[0] - a[0]))) : 0
      const bx = a.slice(1).map((v: number, j: number) => v + (b[j + 1] - v) * k)
      const viol = active.find(f => !(f.box[2] < bx[0] || f.box[0] > bx[2] || f.box[3] < bx[1] || f.box[1] > bx[3]))
      if (opt.onlyViol && !viol) { hide(); continue }
      const g = ground((bx[0] + bx[2]) / 2, bx[3])
      const j1 = Math.max(0, i - 4), j2 = Math.min(p.length - 1, i + 4)
      const d = ground((p[j2][1] + p[j2][3]) / 2, p[j2][4]).sub(ground((p[j1][1] + p[j1][3]) / 2, p[j1][4]))
      if (d.length() > 0.8) heading.set(tr.id, Math.atan2(d.x, d.z))
      const yaw = heading.get(tr.id) ?? Math.PI
      const L = (DIM[tr.type] || DIM.car)[1]
      // the box bottom is the vehicle's nearest edge to the camera: move the centre half a length away from it
      const away = new THREE.Vector3(g.x, 0, g.z).normalize().multiplyScalar(L / 2)
      m.position.set(g.x + away.x * 0.6, 0, g.z + away.z); m.rotation.y = yaw; m.visible = true
      ring.visible = !!viol; if (viol) { ring.position.set(m.position.x, 0.04, m.position.z); (ring.material as THREE.MeshBasicMaterial).color.setHex(ROUTE_COL[viol.route]) }
      const text = viol ? viol.violation : tr.plate || ''
      let lab = labels.get(tr.id)
      if (opt.labels && text) {
        if (!lab || lab.text !== text) {
          if (lab) scene.remove(lab.s)
          lab = { s: spriteLabel(text, viol ? '#' + ROUTE_COL[viol.route].toString(16).padStart(6, '0') : 'rgba(15,31,56,.85)'), text }
          scene.add(lab.s); labels.set(tr.id, lab)
        }
        lab.s.visible = true; lab.s.position.set(m.position.x, (DIM[tr.type]?.[2] || 1.5) + 1.3, m.position.z)
      } else if (lab) lab.s.visible = false
    }
  }, [t, opt, data, gen])

  return <div ref={host} style={{ width: '100%', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)', background: '#dbe7f4' }} />
}
