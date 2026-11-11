// Leadhaul promo: renders venture/sim/convoy_sim.py output frame by frame (window.renderFrame(i)).
// Sim coordinates: x east, y north, z up (metres). Three.js: X = x, Y = z, Z = -y.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { STORY, TOTAL_FRAMES, FPS } from './story.js';

const W = 1920, H = 1080;
const canvas = document.getElementById('c');
const Q = new URLSearchParams(location.search);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: Q.get('aa') !== '0', preserveDrawingBuffer: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.6;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = Q.get('soft') === '0' ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, W / H, 0.5, 12000);
const V = (x, y, z) => new THREE.Vector3(x, z, -y);   // sim -> three

// ------------------------------------------------------------------------------------------------ data
async function getJSON(u) { return (await fetch(u)).json(); }
async function getBin(u) { return (await fetch(u)).arrayBuffer(); }
const [SC, TR, MET, HB, CB] = await Promise.all([
  getJSON('data/scene.json'), getJSON('data/traj.json'), getJSON('data/metrics.json'),
  getBin('data/terrain_height.bin'), getBin('data/terrain_colour.bin')]);
const XS = SC.grid.xs, YS = SC.grid.ys, NX = XS.length, NY = YS.length;
const HGT = new Float32Array(HB), COL = new Uint8Array(CB);
const FR = TR.frames, NF = FR.length, SFPS = TR.fps;
const WB = SC.truck.wheelbase;

function bsearch(a, v) { if (v <= a[0]) return 0; let lo = 0, hi = a.length - 2; if (v >= a[hi]) return hi;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (a[m] <= v) lo = m; else hi = m - 1; } return lo; }
function hAt(x, y) {
  const i = bsearch(XS, x), j = bsearch(YS, y);
  const tx = Math.min(1, Math.max(0, (x - XS[i]) / (XS[i + 1] - XS[i]))), ty = Math.min(1, Math.max(0, (y - YS[j]) / (YS[j + 1] - YS[j])));
  const a = HGT[j * NX + i], b = HGT[j * NX + i + 1], c = HGT[(j + 1) * NX + i], d = HGT[(j + 1) * NX + i + 1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
function angLerp(a, b, t) { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; }

// interpolated state at sim time t: rows [x,y,yaw,steer,odo,v,a,brake,tip,payload,lat,gap,mode,det,detR,z,pitch,roll]
function frameAt(t) { const f = clamp(t * SFPS, 0, NF - 1.001); const i = Math.floor(f); return [i, f - i]; }
function truckAt(t, c, k) {
  const [i, u] = frameAt(t); const A = FR[i].c[c].trucks[k], B = FR[i + 1].c[c].trucks[k];
  const r = A.slice();
  for (const j of [0, 1, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17]) r[j] = lerp(A[j], B[j], u);
  r[2] = angLerp(A[2], B[2], u);
  return r;
}
function convoyAt(t, c) { const [i] = frameAt(t); return FR[i].c[c]; }
function lvAt(t) { const [i, u] = frameAt(t); const A = FR[i].lv, B = FR[i + 1].lv; return A.map((v, j) => j === 2 ? angLerp(A[j], B[j], u) : lerp(v, B[j], u)); }
function frontOf(r) { return [r[0] + WB * Math.cos(r[2]), r[1] + WB * Math.sin(r[2])]; }
function headingSmooth(t, c, k, span = 1.5) { let sx = 0, sy = 0; for (let d = -span; d <= span + 1e-6; d += 0.5) { const r = truckAt(t + d, c, k); sx += Math.cos(r[2]); sy += Math.sin(r[2]); } return Math.atan2(sy, sx); }

// ------------------------------------------------------------------------------------------------ sky, light
const SUN_EL = THREE.MathUtils.degToRad(23), SUN_AZ = THREE.MathUtils.degToRad(218);   // azimuth from east, CCW (south-west)
const sunDirSim = [Math.cos(SUN_EL) * Math.cos(SUN_AZ), Math.cos(SUN_EL) * Math.sin(SUN_AZ), Math.sin(SUN_EL)];
const sunDir = V(...sunDirSim).normalize();
const sky = new Sky(); sky.scale.setScalar(30000);
const su = sky.material.uniforms;
su.turbidity.value = 6.5; su.rayleigh.value = 1.35; su.mieCoefficient.value = 0.006; su.mieDirectionalG.value = 0.82;
su.sunPosition.value.copy(sunDir);
{ const skyScene = new THREE.Scene(); skyScene.add(sky); const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(skyScene, 0.02).texture; skyScene.remove(sky); }
scene.add(sky);
scene.environmentIntensity = 0.32;
scene.fog = new THREE.FogExp2(0xcfc0a2, 0.0004);

const sun = new THREE.DirectionalLight(0xffe2bd, 3.6);
sun.castShadow = true;
const SMS = Number(Q.get('sm') || 4096); sun.shadow.mapSize.set(SMS, SMS);
sun.shadow.bias = -0.00025; sun.shadow.normalBias = 0.06;
const SH = 140; Object.assign(sun.shadow.camera, { left: -SH, right: SH, top: SH, bottom: -SH, near: 10, far: 1600 });
scene.add(sun); scene.add(sun.target);
scene.add(new THREE.HemisphereLight(0xe9e2d6, 0x8a7052, 0.42));
function aimSun(focus) { sun.target.position.copy(focus); sun.position.copy(focus).addScaledVector(sunDir, 700); sun.target.updateMatrixWorld(); }

// ------------------------------------------------------------------------------------------------ textures
function noiseTex(size, seed, contrast, base) {
  const c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d');
  const img = g.createImageData(size, size); const R = rng(seed);
  const n = new Float32Array(size * size); for (let i = 0; i < n.length; i++) n[i] = R();
  for (let pass = 0; pass < 2; pass++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x; n[i] = (n[i] * 2 + n[y * size + ((x + 1) % size)] + n[((y + 1) % size) * size + x]) / 4; }
  for (let i = 0; i < n.length; i++) { const R2 = R(); const v = clamp(base + contrast * (n[i] - 0.5) + 0.12 * (R2 - 0.5), 0, 1) * 255;
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255; }
  g.putImageData(img, 0, 0); const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; return t;
}
const grain = noiseTex(512, 7, 0.9, 0.93);
function textTex(text, opts = {}) {
  const { w = 1024, h = 256, font = '700 150px Inter', color = '#ffffff', bg = null, align = 'center', stroke = null } = opts;
  const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d');
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, w, h); }
  g.font = font; g.fillStyle = color; g.textAlign = align; g.textBaseline = 'middle';
  if (stroke) { g.lineWidth = 8; g.strokeStyle = stroke; g.strokeText(text, align === 'center' ? w / 2 : 20, h / 2); }
  g.fillText(text, align === 'center' ? w / 2 : 20, h / 2);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
}

// ------------------------------------------------------------------------------------------------ terrain
{
  const pos = new Float32Array(NX * NY * 3), col = new Float32Array(NX * NY * 3), uv = new Float32Array(NX * NY * 2);
  const cc = new THREE.Color();
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
    const k = j * NX + i; pos[3 * k] = XS[i]; pos[3 * k + 1] = HGT[k]; pos[3 * k + 2] = -YS[j];
    cc.setRGB(COL[3 * k] / 255, COL[3 * k + 1] / 255, COL[3 * k + 2] / 255, THREE.SRGBColorSpace);
    col[3 * k] = cc.r; col[3 * k + 1] = cc.g; col[3 * k + 2] = cc.b; uv[2 * k] = XS[i] / 9; uv[2 * k + 1] = YS[j] / 9;
  }
  const idx = new Uint32Array((NX - 1) * (NY - 1) * 6); let p = 0;
  for (let j = 0; j < NY - 1; j++) for (let i = 0; i < NX - 1; i++) {
    const a = j * NX + i, b = a + 1, c = a + NX, d = c + 1; idx[p++] = a; idx[p++] = b; idx[p++] = c; idx[p++] = b; idx[p++] = d; idx[p++] = c; }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setIndex(new THREE.BufferAttribute(idx, 1)); g.computeVertexNormals();
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, map: grain, bumpMap: grain, bumpScale: 0.6, roughness: 0.96, metalness: 0.0 });
  const terrain = new THREE.Mesh(g, m); terrain.receiveShadow = true; terrain.castShadow = true; scene.add(terrain);
}

// ------------------------------------------------------------------------------------------------ materials
const M = {
  yellow: new THREE.MeshStandardMaterial({ color: 0xe2a019, roughness: 0.5, metalness: 0.15 }),
  yellowDirty: new THREE.MeshStandardMaterial({ color: 0xc9922a, roughness: 0.75, metalness: 0.1 }),
  steel: new THREE.MeshStandardMaterial({ color: 0x2d3034, roughness: 0.65, metalness: 0.45 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x1b1d20, roughness: 0.8, metalness: 0.2 }),
  tire: new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.92, metalness: 0.0 }),
  rim: new THREE.MeshStandardMaterial({ color: 0xb99628, roughness: 0.5, metalness: 0.3 }),
  glass: new THREE.MeshStandardMaterial({ color: 0x1d2a33, roughness: 0.08, metalness: 0.7 }),
  white: new THREE.MeshStandardMaterial({ color: 0xeeeeea, roughness: 0.45, metalness: 0.1 }),
  teal: new THREE.MeshStandardMaterial({ color: 0x12b8aa, roughness: 0.4, metalness: 0.1, emissive: 0x0a6a62, emissiveIntensity: 0.25 }),
  amber: new THREE.MeshStandardMaterial({ color: 0xff9f1a, roughness: 0.4, metalness: 0.1, emissive: 0x5a3000, emissiveIntensity: 0.25 }),
  rock: new THREE.MeshStandardMaterial({ color: 0x8a7f70, roughness: 0.95, metalness: 0.0, map: grain, flatShading: true }),
  rockLight: new THREE.MeshStandardMaterial({ color: 0xb3a892, roughness: 0.95, metalness: 0.0, map: grain }),
  galv: new THREE.MeshStandardMaterial({ color: 0x9aa1a6, roughness: 0.45, metalness: 0.6 }),
  concrete: new THREE.MeshStandardMaterial({ color: 0xb8b2a6, roughness: 0.9, metalness: 0.0, map: grain }),
  shrub: new THREE.MeshStandardMaterial({ color: 0x5f6236, roughness: 1.0, metalness: 0.0, flatShading: true }),
  head: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff3d6, emissiveIntensity: 1.6 }),
  red: new THREE.MeshStandardMaterial({ color: 0xcc2b2b, roughness: 0.5 }),
};
const geoBox = new THREE.BoxGeometry(1, 1, 1);
function box(mat, x0, x1, y0, y1, z0, z1, parent) {   // local truck coords: x fwd, y up, z right
  const m = new THREE.Mesh(geoBox, mat); m.scale.set(x1 - x0, y1 - y0, z1 - z0); m.position.set((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  m.castShadow = true; m.receiveShadow = true; parent.add(m); return m;
}
function cyl(mat, r, h, parent, seg = 20) { const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, seg), mat); m.castShadow = true; m.receiveShadow = true; parent.add(m); return m; }

// tyre with a tread texture so rotation reads on screen
const treadTex = (() => { const c = document.createElement('canvas'); c.width = 512; c.height = 64; const g = c.getContext('2d');
  g.fillStyle = '#1a1a1a'; g.fillRect(0, 0, 512, 64); g.fillStyle = '#0c0c0c';
  for (let i = 0; i < 32; i++) { g.beginPath(); const x = i * 16; g.moveTo(x, 0); g.lineTo(x + 7, 0); g.lineTo(x + 12, 32); g.lineTo(x + 7, 64); g.lineTo(x, 64); g.lineTo(x + 5, 32); g.fill(); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; })();
const tireMat = [new THREE.MeshStandardMaterial({ map: treadTex, roughness: 0.95 }), M.tire, M.tire];
const tireGeo = new THREE.CylinderGeometry(1.12, 1.12, 0.78, 32, 1); tireGeo.rotateX(Math.PI / 2);
const rimGeo = new THREE.CylinderGeometry(0.62, 0.62, 0.82, 18); rimGeo.rotateX(Math.PI / 2);
const hubGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.9, 10); hubGeo.rotateX(Math.PI / 2);
function wheel(parent, x, y, z) {
  const g = new THREE.Group(); g.position.set(x, y, z); parent.add(g);
  const t = new THREE.Mesh(tireGeo, tireMat); t.castShadow = true; g.add(t);
  const r = new THREE.Mesh(rimGeo, M.rim); g.add(r);
  const hb = new THREE.Mesh(hubGeo, M.steel); g.add(hb);
  for (let i = 0; i < 6; i++) { const b = new THREE.Mesh(geoBox, M.steel); const a = i * Math.PI / 3; b.scale.set(0.12, 0.12, 0.86); b.position.set(0.42 * Math.cos(a), 0.42 * Math.sin(a), 0); g.add(b); }
  return g;
}

// ------------------------------------------------------------------------------------------------ haul truck
const HINGE = new THREE.Vector3(-2.35, 2.05, 0);
function makeTruck(name, role, loaded) {
  const root = new THREE.Group(); root.rotation.order = 'YZX';
  const T = { root, name, role, wheelsF: [], wheelsR: [], steerF: [] };
  // frame and running gear
  box(M.steel, -2.9, 6.2, 0.85, 1.75, -0.7, 0.7, root);
  box(M.steel, -0.6, 0.6, 0.6, 1.45, -1.6, 1.6, root);                  // rear axle housing
  box(M.steel, 3.7, 4.7, 0.7, 1.5, -1.8, 1.8, root);                    // front suspension
  for (const s of [-1, 1]) {
    const sp = new THREE.Group(); sp.position.set(WB, 1.12, s * 1.95); root.add(sp); T.steerF.push(sp);
    T.wheelsF.push(wheel(sp, 0, 0, 0));
    T.wheelsR.push(wheel(root, 0, 1.12, s * 1.58)); T.wheelsR.push(wheel(root, 0, 1.12, s * 2.38));
  }
  // deck, cab (left = -z), hood, grille, bumper
  box(M.yellow, 4.2, 6.95, 2.55, 2.8, -2.55, 2.55, root);
  box(M.yellow, 4.3, 6.6, 1.7, 3.9, -0.85, 0.85, root);
  box(M.dark, 6.6, 6.85, 1.5, 3.75, -1.05, 1.05, root);
  for (let y = 1.7; y < 3.7; y += 0.26) box(M.steel, 6.86, 6.9, y, y + 0.1, -1.0, 1.0, root);
  box(M.yellowDirty, 6.4, 7.05, 0.95, 1.55, -1.9, 1.9, root);
  box(M.yellow, 4.55, 6.35, 2.8, 4.55, -2.5, -0.95, root);              // cab
  box(M.glass, 6.36, 6.4, 3.25, 4.45, -2.42, -1.03, root);
  box(M.glass, 4.75, 6.2, 3.25, 4.45, -2.52, -2.49, root);
  box(M.glass, 4.75, 6.2, 3.25, 4.45, -0.94, -0.91, root);
  box(M.yellow, 4.2, 6.9, 3.55, 3.62, 2.45, 2.55, root);                // handrails
  box(M.yellow, 6.85, 6.92, 2.8, 3.6, -2.5, 2.5, root);
  for (const z of [-2.3, -1.2, 1.2, 2.3]) box(M.yellow, 6.84, 6.9, 2.8, 3.6, z - 0.04, z + 0.04, root);
  const ladder = box(M.steel, 5.6, 7.3, 0.6, 0.72, -2.6, -2.2, root); ladder.rotation.z = 0.62; ladder.position.set(6.6, 1.7, -2.4);
  for (const s of [-1, 1]) { box(M.head, 6.96, 7.06, 2.32, 2.5, s * 1.45 - 0.2, s * 1.45 + 0.2, root); box(M.head, 7.0, 7.07, 1.15, 1.32, s * 1.4 - 0.18, s * 1.4 + 0.18, root); }
  box(M.dark, -1.2, 2.4, 1.2, 1.75, 1.0, 2.0, root);                      // fuel tank
  box(M.dark, -1.2, 2.4, 1.2, 1.75, -2.0, -1.0, root);                    // hydraulic tank
  // tail / brake lights
  const brakeMat = new THREE.MeshStandardMaterial({ color: 0x3a0000, emissive: 0xff1c1c, emissiveIntensity: 0.4, roughness: 0.4 });
  for (const s of [-1, 1]) box(brakeMat, -2.98, -2.9, 1.25, 1.55, s * 0.65 - 0.25, s * 0.65 + 0.25, root);
  T.brakeMat = brakeMat;
  // dump body on a hinge
  const body = new THREE.Group(); body.position.copy(HINGE); root.add(body); T.body = body;
  const bodyPart = (mat, x0, x1, y0, y1, z0, z1) => box(mat, x0 - HINGE.x, x1 - HINGE.x, y0 - HINGE.y, y1 - HINGE.y, z0, z1, body);
  bodyPart(M.yellow, -2.85, 3.55, 2.05, 2.3, -2.5, 2.5);                 // floor
  const tail = bodyPart(M.yellow, -3.55, -2.8, 2.05, 2.3, -2.5, 2.5); tail.rotation.z = -0.45; tail.position.y += 0.22;
  const sideShape = new THREE.Shape([new THREE.Vector2(-3.5, 2.6), new THREE.Vector2(-2.85, 2.15), new THREE.Vector2(3.6, 2.15), new THREE.Vector2(3.6, 4.5), new THREE.Vector2(-3.5, 3.9)].map(v => v.sub(new THREE.Vector2(HINGE.x, HINGE.y))));
  const sideGeo = new THREE.ExtrudeGeometry(sideShape, { depth: 0.16, bevelEnabled: false });
  for (const s of [-1, 1]) { const m = new THREE.Mesh(sideGeo, M.yellow); m.position.z = s > 0 ? 2.38 : -2.54; m.castShadow = m.receiveShadow = true; body.add(m);
    for (const x of [-2.2, -0.6, 1.0, 2.6]) bodyPart(M.yellowDirty, x - 0.09, x + 0.09, 2.2, 4.35 - (3.6 - x) * 0.08, s * 2.54 - 0.05, s * 2.54 + 0.08);
    bodyPart(role === 'lead' ? M.amber : M.teal, -3.35, 3.55, 3.05, 3.3, s * 2.57 - 0.03, s * 2.57 + 0.03); }
  bodyPart(M.yellow, 3.55, 3.8, 2.15, 4.8, -2.55, 2.55);                // front wall
  bodyPart(M.yellow, 3.6, 7.0, 4.78, 4.98, -2.6, 2.6);                  // canopy
  bodyPart(M.yellowDirty, 6.9, 7.0, 4.5, 4.98, -2.6, 2.6);
  // side decal
  const decal = new THREE.MeshBasicMaterial({ map: textTex(role === 'lead' ? 'LEAD' : name, { font: '800 150px Inter', color: '#0b1014' }), transparent: true });
  for (const s of [-1, 1]) { const p = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 1.05), decal); p.position.set(0.1 - HINGE.x, 2.7 - HINGE.y, s * 2.61); if (s < 0) p.rotation.y = Math.PI; body.add(p); }
  // payload heap
  const heapGeo = new THREE.SphereGeometry(1, 28, 12, 0, Math.PI * 2, 0, Math.PI / 2);
  { const p = heapGeo.attributes.position; const R = rng(name.length * 31 + (role === 'lead' ? 7 : 3));
    for (let i = 0; i < p.count; i++) { const y = p.getY(i); const n = 0.82 + 0.3 * R(); p.setXYZ(i, p.getX(i) * (0.9 + 0.12 * R()), y * n, p.getZ(i) * (0.9 + 0.12 * R())); } heapGeo.computeVertexNormals(); }
  const heap = new THREE.Mesh(heapGeo, M.rock); heap.scale.set(3.1, 1.5, 2.25); heap.position.set(0.25 - HINGE.x, 2.45 - HINGE.y, 0); heap.castShadow = true; body.add(heap);
  T.heap = heap; heap.visible = loaded;
  // hoist cylinders
  T.hoists = [-1, 1].map(s => { const c = cyl(M.galv, 0.16, 1, root, 10); c.userData.base = new THREE.Vector3(1.3, 1.45, s * 1.0); c.userData.att = new THREE.Vector3(1.7 - HINGE.x, 2.05 - HINGE.y, s * 1.0); return c; });
  // autonomy kit (followers) / lead kit
  const kit = new THREE.Group(); root.add(kit);
  const v2v = cyl(M.dark, 0.035, 0.9, kit, 6); v2v.position.set(6.0, 5.45, 0.9);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), M.white); tip.position.set(6.0, 5.92, 0.9); kit.add(tip);
  for (const z of [-1.9, 1.9]) { const gn = cyl(M.white, 0.17, 0.08, kit, 14); gn.position.set(4.6, 5.04, z); }
  if (role === 'follow') {
    for (const z of [-2.35, 2.35]) { const l = cyl(M.dark, 0.14, 0.2, kit, 16); l.position.set(6.82, 2.92, z); const d = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 6, 0, Math.PI * 2, 0, Math.PI / 2), M.glass); d.position.set(6.82, 3.02, z); kit.add(d); }
    const top = cyl(M.dark, 0.16, 0.22, kit, 16); top.position.set(6.55, 5.1, 0);
    box(M.dark, 7.03, 7.15, 1.12, 1.42, -0.32, 0.32, kit);              // radar
    box(M.dark, 6.75, 7.02, 4.62, 4.76, -0.5, 0.5, kit);                // camera bar
  }
  const statusMat = new THREE.MeshStandardMaterial({ color: 0x111111, emissive: role === 'lead' ? 0xffa21a : 0x22ff9a, emissiveIntensity: 2.0 });
  if (role === 'lead') { const b = cyl(statusMat, 0.17, 0.28, root, 14); b.position.set(5.3, 5.13, -1.2); }
  else box(statusMat, 6.3, 6.5, 4.98, 5.1, -0.9, 0.9, root);
  T.statusMat = statusMat;
  root.traverse(o => { if (o.isMesh) { o.castShadow = true; } });
  scene.add(root);
  return T;
}

const NAMES = ['LEAD', 'AUTO 1', 'AUTO 2', 'AUTO 3'];
const fleets = [0, 1].map(c => NAMES.map((nm, k) => makeTruck(nm, k === 0 ? 'lead' : 'follow', c === 0)));
const _q = new THREE.Quaternion(), _up = new THREE.Vector3(0, 1, 0), _v = new THREE.Vector3(), _a = new THREE.Vector3();
function poseTruck(T, r, t, convoyState) {
  T.root.position.copy(V(r[0], r[1], r[15]));
  T.root.rotation.set(r[17], r[2], r[16]);
  for (const s of T.steerF) s.rotation.y = r[3];
  const ang = -r[4] / 1.12; for (const w of [...T.wheelsF, ...T.wheelsR]) w.rotation.z = ang;
  T.body.rotation.z = r[8];
  const pay = r[9];
  T.heap.visible = pay > 0.02; T.heap.scale.set(3.1 * (0.55 + 0.45 * pay), 1.5 * pay, 2.25 * (0.75 + 0.25 * pay)); T.heap.position.x = 0.25 - HINGE.x - (1 - pay) * 2.2;
  T.brakeMat.emissiveIntensity = 0.35 + 3.5 * r[7];
  const hold = convoyState === 'HOLD' || convoyState === 'CLEAR';
  if (T.role === 'lead') T.statusMat.emissiveIntensity = 1.2 + 1.8 * Math.max(0, Math.sin(t * 9));
  else { T.statusMat.emissive.setHex(hold ? 0xff2a2a : 0x22ff9a); T.statusMat.emissiveIntensity = hold ? 1.0 + 2.2 * (Math.sin(t * 12) > 0 ? 1 : 0) : 1.6; }
  for (const c of T.hoists) {
    const base = c.userData.base; _a.copy(c.userData.att).applyAxisAngle(new THREE.Vector3(0, 0, 1), r[8]).add(HINGE);
    _v.subVectors(_a, base); const len = _v.length(); c.position.copy(base).addScaledVector(_v, 0.5); c.scale.set(1, len, 1); c.quaternion.setFromUnitVectors(_up, _v.normalize());
  }
}

// ------------------------------------------------------------------------------------------------ light vehicle
const pickup = (() => {
  const g = new THREE.Group(); g.rotation.order = 'YZX';
  box(M.white, -2.8, 2.8, 0.55, 1.15, -1.0, 1.0, g);
  box(M.white, -0.2, 1.75, 1.15, 1.95, -0.98, 0.98, g);
  box(M.glass, 1.75, 1.78, 1.25, 1.85, -0.9, 0.9, g); box(M.glass, 0.0, 1.6, 1.3, 1.85, -1.0, -0.98, g); box(M.glass, 0.0, 1.6, 1.3, 1.85, 0.98, 1.0, g);
  box(M.white, -2.8, -0.25, 1.15, 1.45, -1.0, -0.92, g); box(M.white, -2.8, -0.25, 1.15, 1.45, 0.92, 1.0, g); box(M.white, -2.85, -2.75, 1.15, 1.45, -1.0, 1.0, g);
  box(M.amber, 0.4, 1.1, 1.95, 2.08, -0.75, 0.75, g);
  box(M.dark, 2.75, 2.9, 0.5, 0.85, -0.95, 0.95, g);
  const ws = []; for (const x of [-1.75, 1.75]) for (const s of [-1, 1]) { const w = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.3, 18).rotateX(Math.PI / 2), M.tire); w.position.set(x, 0.42, s * 0.88); w.castShadow = true; g.add(w); ws.push(w); }
  const pole = cyl(M.dark, 0.02, 3.6, g, 6); pole.position.set(-2.6, 3.0, 0.85);
  const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.55, 0.32), new THREE.MeshStandardMaterial({ color: 0xff5a00, side: THREE.DoubleSide, emissive: 0x401000 })); flag.position.set(-2.33, 4.62, 0.85); g.add(flag);
  scene.add(g); return { g, ws };
})();

// ------------------------------------------------------------------------------------------------ excavator at the face
const exc = (() => {
  const x = 30, y = 56, z = hAt(x, y);
  const root = new THREE.Group(); root.position.copy(V(x, y, z)); root.rotation.y = THREE.MathUtils.degToRad(200); scene.add(root);
  for (const s of [-1, 1]) { box(M.dark, -3.6, 3.6, 0, 1.3, s * 1.9 - 0.55, s * 1.9 + 0.55, root); }
  const house = new THREE.Group(); house.position.y = 1.5; root.add(house);
  box(M.yellow, -3.2, 2.0, 0.1, 2.6, -1.6, 1.6, house); box(M.dark, -3.4, -2.2, 0.1, 2.3, -1.7, 1.7, house);
  box(M.yellow, 0.6, 2.4, 2.6, 4.4, -1.6, -0.3, house); box(M.glass, 2.4, 2.43, 2.8, 4.2, -1.5, -0.4, house);
  const boom = new THREE.Group(); boom.position.set(1.8, 2.2, 0.6); house.add(boom);
  box(M.yellow, 0, 7.5, -0.5, 0.5, -0.45, 0.45, boom);
  const stick = new THREE.Group(); stick.position.set(7.4, 0, 0); boom.add(stick);
  box(M.yellow, 0, 4.6, -0.38, 0.38, -0.35, 0.35, stick);
  const bucket = new THREE.Group(); bucket.position.set(4.5, 0, 0); stick.add(bucket);
  box(M.steel, -0.3, 1.0, -1.6, 0.2, -1.1, 1.1, bucket);
  root.traverse(o => { if (o.isMesh) o.castShadow = true; });
  return { house, boom, stick, bucket };
})();
function animateExcavator(t) {
  const c = 0.5 + 0.5 * Math.sin(t * 0.42);
  exc.house.rotation.y = lerp(-0.2, 1.4, c);
  exc.boom.rotation.z = lerp(0.62, 0.2, smooth(0.2, 0.8, c));
  exc.stick.rotation.z = lerp(-1.75, -1.05, c);
  exc.bucket.rotation.z = lerp(-1.2, 0.2, smooth(0.6, 1.0, c));
}
// blasted rock at the face
function lumpy(radius, height, seed, mat, segs = 28) {
  const g = new THREE.SphereGeometry(1, segs, 10, 0, Math.PI * 2, 0, Math.PI / 2); const p = g.attributes.position; const R = rng(seed);
  for (let i = 0; i < p.count; i++) { const k = 0.82 + 0.3 * R(); p.setXYZ(i, p.getX(i) * k, p.getY(i) * (0.85 + 0.3 * R()), p.getZ(i) * k); }
  g.computeVertexNormals(); const m = new THREE.Mesh(g, mat); m.scale.set(radius, height, radius); m.castShadow = m.receiveShadow = true; return m;
}
{ const m = lumpy(22, 7, 5, M.rock, 40); m.position.copy(V(52, 70, hAt(52, 70) - 1.5)); m.scale.z = 14; scene.add(m); }

// ------------------------------------------------------------------------------------------------ plant
{
  const zp = SC.z_pad;
  const P = (x, y) => V(x, y, zp);
  const blk = (mat, cx, cy, w, d, h, z0 = 0, rot = 0) => { const m = new THREE.Mesh(geoBox, mat); m.scale.set(w, h, d); m.position.copy(P(cx, cy)); m.position.y += z0 + h / 2; m.rotation.y = rot; m.castShadow = m.receiveShadow = true; scene.add(m); return m; };
  blk(M.galv, 612, 305, 20, 16, 15); blk(M.yellow, 612, 296, 14, 4, 2.5, 13); blk(M.dark, 612, 296, 10, 3, 3, 9);       // primary crusher + dump pocket
  blk(M.galv, 665, 322, 14, 12, 20); blk(M.dark, 665, 322, 15, 13, 1.2, 20);                                             // screen tower
  for (const [x, y] of [[716, 300], [726, 300], [736, 300]]) { const s = cyl(M.galv, 3.6, 18, scene, 28); s.position.copy(P(x, y)); s.position.y += 9; const c = new THREE.Mesh(new THREE.ConeGeometry(3.6, 2.5, 28), M.galv); c.position.copy(P(x, y)); c.position.y += 19.2; scene.add(c); }
  const conv = (x0, y0, z0, x1, y1, z1) => { const a = V(x0, y0, zp + z0), b = V(x1, y1, zp + z1); const d = new THREE.Vector3().subVectors(b, a); const m = new THREE.Mesh(geoBox, M.galv);
    m.scale.set(d.length(), 0.9, 1.6); m.position.copy(a).addScaledVector(d, 0.5); m.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), d.clone().normalize()); m.castShadow = true; scene.add(m);
    for (let u = 0.15; u < 1; u += 0.22) { const p = a.clone().addScaledVector(d, u); const h = p.y - V(0, 0, zp).y; const leg = new THREE.Mesh(geoBox, M.steel); leg.scale.set(0.4, h, 0.4); leg.position.set(p.x, p.y - h / 2, p.z); leg.castShadow = true; scene.add(leg); } };
  conv(622, 312, 6, 658, 318, 18); conv(672, 328, 18, 700, 352, 26); conv(658, 330, 14, 628, 352, 22); conv(672, 316, 12, 712, 318, 14);
  for (const [x, y, r, h, mat] of [[703, 356, 19, 15, M.rockLight], [626, 356, 15, 12, M.rockLight], [745, 330, 13, 9, M.rock], [690, 380, 17, 11, M.rockLight]]) { const m = lumpy(r, h, x, mat, 36); m.position.copy(P(x, y)); scene.add(m); }
  blk(M.white, 590, 345, 22, 12, 5); blk(M.dark, 590, 345, 23, 13, 0.4, 5); blk(M.galv, 560, 330, 26, 16, 8); blk(M.steel, 560, 330, 27, 17, 0.5, 8);  // office, workshop
  for (const [x, y] of [[640, 288], [560, 290], [700, 176], [552, 176]]) { const p = cyl(M.galv, 0.18, 18, scene, 8); p.position.copy(P(x, y)); p.position.y += 9; blk(M.white, x, y, 1.6, 0.6, 0.8, 18); }
  // older dump piles on the pad
  const R = rng(99);
  for (let k = 0; k < 14; k++) { const x = 566 + (k % 7) * 12 + 4 * R(), y = 268 - Math.floor(k / 7) * 15 + 3 * R(); const m = lumpy(3.8 + R(), 2.2 + 0.6 * R(), 200 + k, M.rock); m.position.copy(V(x, y, hAt(x, y) - 0.2)); scene.add(m); }
}

// ------------------------------------------------------------------------------------------------ roadside: delineators, signs, poles
const ROAD = SC.road.xy, ROADZ = SC.road.z, RH = SC.road.head;
function roadDist(x, y) { let best = 1e9; for (let i = 0; i < ROAD.length; i += 2) { const dx = ROAD[i][0] - x, dy = ROAD[i][1] - y; const d = dx * dx + dy * dy; if (d < best) best = d; } return Math.sqrt(best); }
{
  const post = new THREE.InstancedMesh(new THREE.BoxGeometry(0.16, 1.2, 0.16), M.white, 200); let n = 0; const mtx = new THREE.Matrix4();
  for (let i = 0; i < ROAD.length; i += 8) {
    const s = SC.road.s[i]; if (s < SC.road.ramp[0] + 10 || Math.abs(s - SC.road.s_cross) < 22 || s > SC.road.s[SC.road.s.length - 1] - 20) continue;
    for (const side of [-1, 1]) { const nx = -Math.sin(RH[i]) * side, ny = Math.cos(RH[i]) * side; const x = ROAD[i][0] + nx * 16.5, y = ROAD[i][1] + ny * 16.5;
      mtx.makeTranslation(...V(x, y, hAt(x, y) + 0.6).toArray()); post.setMatrixAt(n++, mtx); }
  }
  post.count = n; post.castShadow = true; scene.add(post);
  const sign = (x, y, yawDeg, tex, w, h, hpole = 3.2) => { const z = hAt(x, y); const g = new THREE.Group(); g.position.copy(V(x, y, z)); g.rotation.y = THREE.MathUtils.degToRad(yawDeg); scene.add(g);
    const p = cyl(M.galv, 0.06, hpole + h / 2, g, 8); p.position.y = (hpole + h / 2) / 2;
    const b = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 })); b.position.set(0, hpole, 0.08); g.add(b);
    const bk = new THREE.Mesh(new THREE.PlaneGeometry(w, h), M.galv); bk.position.set(0, hpole, 0.06); bk.rotation.y = Math.PI; g.add(bk); };
  // yaw: the sign face points to sim heading (sin yaw, -cos yaw); 180 = faces north, 0 = south, -90 = west
  sign(372, 151, 180, textTex('STOP', { w: 512, h: 512, font: '800 170px Inter', bg: '#c8102e' }), 1.4, 1.4);
  sign(388, 110, 0, textTex('YIELD TO HAUL TRUCKS', { w: 1024, h: 256, font: '800 84px Inter', color: '#111', bg: '#ffcc00' }), 3.2, 0.8);
  sign(190, 96, -45, textTex('AUTONOMOUS HAUL ROAD', { w: 1024, h: 256, font: '800 78px Inter', color: '#fff', bg: '#0f6f68' }), 4.6, 1.15, 3.6);
  sign(262, 103, -90, textTex('35 km/h', { w: 512, h: 384, font: '800 140px Inter', color: '#111', bg: '#f2f2f2' }), 1.4, 1.05);
  // power line along the access road
  for (let y = 175; y < 330; y += 38) { const p = cyl(new THREE.MeshStandardMaterial({ color: 0x6b5236, roughness: 0.9 }), 0.16, 11, scene, 8); p.position.copy(V(392, y, hAt(392, y) + 5.5)); const cb = new THREE.Mesh(geoBox, M.dark); cb.scale.set(2.4, 0.15, 0.15); cb.position.copy(V(392, y, hAt(392, y) + 10.5)); scene.add(cb); }
}

// rocks and shrubs (instanced)
{
  const R = rng(4242); const mtx = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s3 = new THREE.Vector3();
  const rockG = new THREE.DodecahedronGeometry(1, 0); const rocks = new THREE.InstancedMesh(rockG, M.rock, 1100); let n = 0;
  const shrubG = new THREE.IcosahedronGeometry(1, 1); const shrubs = new THREE.InstancedMesh(shrubG, M.shrub, 2200); let m = 0;
  const pit = (x, y) => Math.sqrt((x / SC.pit.a) ** 2 + (y / SC.pit.b) ** 2);
  const inBox = (x, y, b, pad = 6) => x > b[0] - pad && x < b[1] + pad && y > b[2] - pad && y < b[3] + pad;
  const clumps = []; for (let i = 0; i < 220; i++) clumps.push([-320 + 1140 * R(), -260 + 720 * R()]);
  for (let tries = 0; tries < 40000 && (n < 1100 || m < 2200); tries++) {
    let x = -320 + 1140 * R(), y = -260 + 720 * R();
    if (R() < 0.8) { const c = clumps[Math.floor(R() * clumps.length)]; x = c[0] + 18 * (R() - 0.5) * 2; y = c[1] + 14 * (R() - 0.5) * 2; }
    const rd = roadDist(x, y); if (rd < 17) continue;
    if (inBox(x, y, SC.pad) || inBox(x, y, SC.plant) || Math.abs(x - SC.access_x) < 9) continue;
    const pr = pit(x, y); const z = hAt(x, y); const slope = Math.hypot(hAt(x + 1, y) - hAt(x - 1, y), hAt(x, y + 1) - hAt(x, y - 1)) / 2;
    const toe = Math.hypot(hAt(x + 4, y) - hAt(x - 4, y), hAt(x, y + 4) - hAt(x, y - 4)) / 8 > 0.25;
    if (n < 1100 && slope < 0.45 && (pr < 1.02 ? R() < (toe ? 0.5 : 0.05) : (rd < 30 ? R() < 0.25 : R() < 0.02))) {
      const sc = pr < 1.02 ? 0.5 + 1.4 * R() ** 2 : 0.4 + 0.9 * R() ** 2; e.set(R() * 6, R() * 6, R() * 6); q.setFromEuler(e); s3.set(sc * (0.8 + 0.5 * R()), sc * (0.6 + 0.4 * R()), sc);
      mtx.compose(V(x, y, z + sc * 0.25), q, s3); rocks.setMatrixAt(n, mtx); rocks.setColorAt(n++, new THREE.Color(pr < 1.02 ? 0xc9c1b2 : 0x9a8e7c));
    } else if (m < 2200 && pr > 1.12 && R() < 0.85) {
      const sc = 0.5 + 0.9 * R(); e.set(0, R() * 6, 0); q.setFromEuler(e); s3.set(sc * 1.3, sc * 0.7, sc * 1.2);
      mtx.compose(V(x, y, z + sc * 0.3), q, s3); shrubs.setMatrixAt(m++, mtx);
    }
  }
  rocks.count = n; shrubs.count = m; rocks.castShadow = shrubs.castShadow = true; rocks.receiveShadow = shrubs.receiveShadow = true; scene.add(rocks); scene.add(shrubs);
}

// ------------------------------------------------------------------------------------------------ dynamic fx: dust, piles, breadcrumbs, corridors
const dustTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d'); const gr = g.createRadialGradient(64, 64, 4, 64, 64, 62);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c); })();
const DUSTN = 1400;
const dustGeo = new THREE.BufferGeometry();
dustGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(DUSTN * 3), 3));
dustGeo.setAttribute('size', new THREE.BufferAttribute(new Float32Array(DUSTN), 1));
dustGeo.setAttribute('alpha', new THREE.BufferAttribute(new Float32Array(DUSTN), 1));
const dustMat = new THREE.ShaderMaterial({
  uniforms: { map: { value: dustTex }, color: { value: new THREE.Color(0xd8c7a3) }, scale: { value: H / (2 * Math.tan(THREE.MathUtils.degToRad(20))) } },
  vertexShader: `attribute float size; attribute float alpha; varying float vA; uniform float scale;
    void main(){ vA = alpha; vec4 mv = modelViewMatrix * vec4(position,1.0); gl_PointSize = clamp(size * scale / -mv.z, 0.0, 600.0); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform sampler2D map; uniform vec3 color; varying float vA; void main(){ vec4 t = texture2D(map, gl_PointCoord); gl_FragColor = vec4(color, t.a * vA); }`,
  transparent: true, depthWrite: false });
const dust = new THREE.Points(dustGeo, dustMat); dust.frustumCulled = false; scene.add(dust);
function updateDust(t) {
  const P = dustGeo.attributes.position.array, S = dustGeo.attributes.size.array, A = dustGeo.attributes.alpha.array; let n = 0;
  const emit = (x, y, v, age, seed, k) => { if (n >= DUSTN) return; const R = rng(seed); R(); const j1 = R() - 0.5, j2 = R() - 0.5;
    const wx = 1.1 * age, wy = 0.7 * age; const px = x + wx + j1 * (1 + age), py = y + wy + j2 * (1 + age); const pz = hAt(px, py) + 0.6 + 0.9 * age * (0.6 + R());
    const vv = clamp((v - 1.5) / 8, 0, 1); const a = 0.30 * vv * Math.pow(1 - age / 3.6, 1.6) * k; if (a < 0.004) return;
    const p = V(px, py, pz); P[3 * n] = p.x; P[3 * n + 1] = p.y; P[3 * n + 2] = p.z; S[n] = 1.6 + 3.2 * age; A[n] = a; n++; };
  for (let c = 0; c < 2; c++) for (let k = 0; k < 4; k++) {
    if (!convoyAt(t, c).active) continue;
    for (let age = 0.0; age < 3.6; age += 0.13) {
      const te = t - age; if (te < 0) break; const r = truckAt(te, c, k); if (r[5] < 1.5) continue; const em = Math.floor(te * 7.7);
      for (const s of [-1, 1]) { const x = r[0] - 1.0 * Math.cos(r[2]) - s * 2.4 * Math.sin(r[2]), y = r[1] - 1.0 * Math.sin(r[2]) + s * 2.4 * Math.cos(r[2]); emit(x, y, r[5], age, em * 131 + c * 17 + k * 5 + (s > 0 ? 1 : 0), 1.0); }
    }
  }
  for (let age = 0; age < 2.6; age += 0.15) { const te = t - age; if (te < 0) break; const l = lvAt(te); if (l[4] < 2) continue; emit(l[0], l[1] + 2.4, l[4], age, Math.floor(te * 6.7) * 7 + 3, 0.6); }
  for (let i = n; i < DUSTN; i++) A[i] = 0;
  dustGeo.setDrawRange(0, n); dustGeo.attributes.position.needsUpdate = dustGeo.attributes.size.needsUpdate = dustGeo.attributes.alpha.needsUpdate = true;
}
// dump piles from the trajectory (where each truck first tips past 0.4 rad)
const piles = [];
for (let k = 0; k < 4; k++) {
  for (let i = 0; i < NF; i++) { const r = FR[i].c[0].trucks[k]; if (r[8] > 0.4) {
    const x = r[0] - 5.2 * Math.cos(r[2]), y = r[1] - 5.2 * Math.sin(r[2]); const m = lumpy(4.6, 2.9, 300 + k, M.rock); m.position.copy(V(x, y, hAt(x, y) - 0.15)); m.visible = false; scene.add(m);
    piles.push({ m, k, i0: i }); break; } }
}
function updatePiles(t) { const [i] = frameAt(t); for (const p of piles) { const pay = FR[i].c[0].trucks[p.k][9]; const g = clamp(1 - pay, 0, 1); p.m.visible = i >= p.i0 && g > 0.02; p.m.scale.set(4.6 * (0.5 + 0.5 * g), 2.9 * g, 4.6 * (0.5 + 0.5 * g)); } }
// leader front-axle breadcrumbs (every 2.5 m of leader travel), per convoy
const crumbs = [0, 1].map(c => { const out = []; let last = -1e9; for (let i = 0; i < NF; i++) { const f = FR[i].c[c]; if (!f.active) continue; const r = f.trucks[0]; if (r[4] - last >= 2.5) { last = r[4]; const [fx, fy] = frontOf(r); out.push({ x: fx, y: fy, z: hAt(fx, fy), i }); } } return out; });
const crumbMesh = new THREE.InstancedMesh(new THREE.CircleGeometry(0.42, 14).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x29f0de, transparent: true, opacity: 0.9, depthWrite: false }), 1200);
crumbMesh.frustumCulled = false; scene.add(crumbMesh);
function nearestCrumb(list, x, y, upto) { let best = 0, bd = 1e18; for (let j = 0; j < list.length && list[j].i <= upto; j++) { const d = (list[j].x - x) ** 2 + (list[j].y - y) ** 2; if (d < bd) { bd = d; best = j; } } return best; }
function updateCrumbs(t, on) {
  if (on <= 0.01) { crumbMesh.count = 0; return; }
  const [fi] = frameAt(t); const mtx = new THREE.Matrix4(); let n = 0;
  for (const c of [0]) { const L = crumbs[c]; const last = truckAt(t, c, 3); const [fx, fy] = frontOf(last); const j0 = nearestCrumb(L, fx, fy, fi);
    for (let j = Math.max(0, j0 - 6); j < L.length && L[j].i <= fi; j++) { const p = L[j]; const fade = smooth(j0 - 6, j0 + 10, j); const s = 0.4 + 0.6 * fade; mtx.makeScale(s, 1, s); mtx.setPosition(V(p.x, p.y, p.z + 0.18)); crumbMesh.setMatrixAt(n++, mtx); } }
  crumbMesh.count = n; crumbMesh.instanceMatrix.needsUpdate = true; crumbMesh.material.opacity = 0.85 * on;
}
// perception corridors: each follower's swept path ahead (80 m), on the leader's breadcrumbs
const corridors = [1, 2, 3].map(() => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(2 * 60 * 3), 3));
  const idx = []; for (let i = 0; i < 59; i++) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } g.setIndex(idx);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x19d3c5, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4 }));
  m.frustumCulled = false; scene.add(m); return m; });
function updateCorridors(t, on) {
  const [fi] = frameAt(t); const L = crumbs[0];
  for (let k = 1; k <= 3; k++) {
    const m = corridors[k - 1]; const r = truckAt(t, 0, k); m.visible = on > 0.01 && r[12] === 0; if (!m.visible) continue;
    const [fx, fy] = frontOf(r); const j0 = nearestCrumb(L, fx, fy, fi); const P = m.geometry.attributes.position.array; let cnt = 0;
    for (let j = j0; j < Math.min(L.length - 1, j0 + 32) && L[j + 1].i <= fi; j++) {
      const p = L[j], q = L[j + 1]; const h = Math.atan2(q.y - p.y, q.x - p.x); const nx = -Math.sin(h) * 4.15, ny = Math.cos(h) * 4.15;
      const a = V(p.x + nx, p.y + ny, hAt(p.x + nx, p.y + ny) + 0.3), b = V(p.x - nx, p.y - ny, hAt(p.x - nx, p.y - ny) + 0.3);
      P.set([a.x, a.y, a.z, b.x, b.y, b.z], 6 * cnt); cnt++; if (cnt >= 60) break; }
    m.geometry.setDrawRange(0, Math.max(0, (cnt - 1) * 6)); m.geometry.attributes.position.needsUpdate = true;
    const det = r[13]; m.material.color.setHex(det === 2 ? 0xff3b3b : det === 1 ? 0xffc21a : 0x19d3c5); m.material.opacity = (det === 2 ? 0.34 : 0.2) * on;
  }
}
const detBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(6.2, 2.6, 2.7)), new THREE.LineBasicMaterial({ color: 0xff3b3b }));
detBox.rotation.order = 'YZX'; scene.add(detBox);

// ------------------------------------------------------------------------------------------------ HUD helpers
const svg = document.getElementById('svg'), chips = document.getElementById('chips'), layers = document.getElementById('layers');
const NS = 'http://www.w3.org/2000/svg';
function project(x, y, z) { const v = V(x, y, z).project(camera); return { x: (v.x + 1) / 2 * W, y: (1 - v.y) / 2 * H, ok: v.z < 1 && v.z > -1 }; }
const chipEls = [0, 1].map(c => NAMES.map((nm, k) => { const d = document.createElement('div'); d.className = 'chip ' + (k === 0 ? 'lead' : 'auto'); chips.appendChild(d); return d; }));
const lvChip = document.createElement('div'); lvChip.className = 'chip hold'; chips.appendChild(lvChip);
for (const id of Object.keys(STORY.layers)) { const d = document.createElement('div'); d.id = 'L_' + id; d.innerHTML = STORY.layers[id](MET); d.style.opacity = 0; layers.appendChild(d); }
const chartBox = document.createElement('div'); chartBox.id = 'chartbox'; chartBox.innerHTML = '<div class="cap">Speed, km/h</div><canvas id="chart" width="600" height="250"></canvas>'; layers.appendChild(chartBox);
const chart = chartBox.querySelector('canvas').getContext('2d');
function drawChart(t, tMark) {
  const g = chart; g.clearRect(0, 0, 600, 250); const x0 = 56, x1 = 580, y0 = 220, y1 = 50, span = 24, vmax = 40;
  g.strokeStyle = 'rgba(255,255,255,0.12)'; g.lineWidth = 1; g.font = '500 13px JetBrains Mono'; g.fillStyle = 'rgba(255,255,255,0.55)';
  for (const v of [0, 10, 20, 30, 40]) { const y = y0 - (y0 - y1) * v / vmax; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); g.fillText(String(v), 22, y + 4); }
  if (tMark !== null && tMark > t - span && tMark <= t) { const x = x0 + (x1 - x0) * (tMark - (t - span)) / span; g.strokeStyle = 'rgba(255,77,77,0.9)'; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(x, y1 - 8); g.lineTo(x, y0); g.stroke(); g.setLineDash([]); g.fillStyle = '#ff4d4d'; g.fillText('HOLD', x > 520 ? x - 42 : x + 6, y1 - 2); }
  const cols = ['#ffb020', '#19d3c5', '#6fe6dc', '#b6f3ee'];
  for (let k = 3; k >= 0; k--) { g.strokeStyle = cols[k]; g.lineWidth = k === 0 ? 3 : 2.4; g.beginPath(); let first = true;
    for (let s = t - span; s <= t + 1e-6; s += 0.2) { if (s < 0) continue; const v = truckAt(s, 0, k)[5] * 3.6; const x = x0 + (x1 - x0) * (s - (t - span)) / span, y = y0 - (y0 - y1) * Math.min(v, vmax) / vmax; if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y); } g.stroke(); }
  g.font = '600 13px Inter'; NAMES.forEach((nm, k) => { g.fillStyle = cols[k]; g.fillRect(250 + k * 84, 18, 14, 4); g.fillText(nm, 268 + k * 84, 24); });
}
const logEl = document.createElement('div'); logEl.id = 'log'; layers.appendChild(logEl);
const EV = (() => { const ev = {}; for (const [t, s] of MET.hud_events) { if (s.includes('in path')) ev.detect = t; if (s.startsWith('Corridor clear')) ev.clear = t; if (s.startsWith('Operator')) ev.resume = t; if (s.includes('approaching')) ev.approach = t; } return ev; })();
function stoppedAt(c, k, after) { for (let i = Math.floor(after * SFPS); i < NF; i++) if (FR[i].c[c].trucks[k][5] < 0.05) return i / SFPS; return null; }
EV.f1stop = stoppedAt(0, 1, EV.detect); EV.allstop = Math.max(...[0, 1, 2, 3].map(k => stoppedAt(0, k, EV.detect)));
function drawLog(t) {
  const rows = [[EV.detect, 0.00, `AUTO 1 · light vehicle in path, ${MET.detection.path_dist_at_detect.toFixed(0)} m`], [EV.detect + 0.05, 0.05, 'V2V hold → LEAD · AUTO 2 · AUTO 3'],
    [EV.f1stop, EV.f1stop - EV.detect, `AUTO 1 stopped · ${MET.closest_approach_to_light_vehicle_m.toFixed(0)} m clear`], [EV.allstop, EV.allstop - EV.detect, 'Convoy holding'],
    [EV.clear, EV.clear - EV.detect, 'Corridor clear · awaiting operator'], [EV.resume, EV.resume - EV.detect, 'Operator taps RESUME']];
  logEl.innerHTML = '<h3>Convoy event log</h3>' + rows.filter(r => t >= r[0]).map(r => `<div class="row"><span class="ts">+${r[1].toFixed(2)} s</span><span>${r[2]}</span></div>`).join('');
}

// ------------------------------------------------------------------------------------------------ frame
const ctx = { truckAt, convoyAt, lvAt, headingSmooth, frontOf, hAt, EV, MET };
let lastShot = null;
window.TOTAL_FRAMES = TOTAL_FRAMES;
window.renderFrame = (fi) => {
  const tv = fi / FPS; const shot = STORY.shots.find(s => tv >= s.t0 && tv < s.t1) || STORY.shots[STORY.shots.length - 1];
  const tau = tv - shot.t0, dur = shot.t1 - shot.t0; const t = shot.sim(tau, dur);
  // world
  for (let c = 0; c < 2; c++) { const cs = convoyAt(t, c); for (let k = 0; k < 4; k++) { const T = fleets[c][k]; T.root.visible = cs.active; if (cs.active) poseTruck(T, truckAt(t, c, k), t, cs.state); } }
  const lv = lvAt(t); pickup.g.position.copy(V(lv[0], lv[1], lv[5])); pickup.g.rotation.set(0, lv[2], lv[6]); for (const w of pickup.ws) w.rotation.z = -lv[3] / 0.42;
  animateExcavator(t); updateDust(t); updatePiles(t);
  const fx = shot.fx || {}; const fxOn = key => (typeof fx[key] === 'function' ? fx[key](tau, dur) : (fx[key] || 0));
  updateCrumbs(t, fxOn('crumbs')); updateCorridors(t, fxOn('corridors'));
  // detection box
  let det = 0, detR = 0; for (let k = 1; k <= 3; k++) { const r = truckAt(t, 0, k); if (r[13] > det) { det = r[13]; detR = r[14]; } }
  detBox.visible = fxOn('detect') > 0 && det > 0; if (detBox.visible) { detBox.position.copy(V(lv[0], lv[1], lv[5] + 1.3)); detBox.rotation.set(0, lv[2], 0); detBox.material.color.setHex(det === 2 ? 0xff3b3b : 0xffc21a); }
  // camera
  const cam = shot.cam(tau, dur, t, ctx); camera.position.copy(V(...cam.pos)); camera.up.set(0, 1, 0); camera.lookAt(V(...cam.target)); camera.fov = cam.fov || 40; camera.updateProjectionMatrix();
  aimSun(V(...(cam.focus || cam.target)));
  renderer.render(scene, camera);
  // HUD: chips
  const chipOn = fxOn('chips');
  for (let c = 0; c < 2; c++) { const cs = convoyAt(t, c); for (let k = 0; k < 4; k++) { const el = chipEls[c][k]; const r = truckAt(t, c, k);
    const show = chipOn > 0.01 && cs.active && (c === 0 || fxOn('chipsReturn') > 0) && r[12] < 3;
    if (!show) { el.style.opacity = 0; continue; }
    const p = project(r[0] + 1.5 * Math.cos(r[2]), r[1] + 1.5 * Math.sin(r[2]), r[15] + 8.2);
    const mk = shot.fx && shot.fx.chipMask; const masked = mk && p.x > mk[0] && p.x < mk[2] && p.y > mk[1] && p.y < mk[3];
    if (!p.ok || masked || p.x < -100 || p.x > W + 100 || p.y < 40 || p.y > H + 50) { el.style.opacity = 0; continue; }
    const hold = cs.state === 'HOLD' || cs.state === 'CLEAR';
    el.className = 'chip ' + (k === 0 ? 'lead' : hold ? 'hold' : 'auto');
    const role = c === 1 ? (k === 0 ? 'return · operator' : 'return · driverless') : (k === 0 ? 'operator' : 'driverless');
    const tele = fxOn('tele') > 0 ? `<span class="tele">${(r[5] * 3.6).toFixed(0).padStart(2)} km/h${k > 0 && r[11] !== null ? ` · gap ${r[11].toFixed(0)} m · ±${Math.abs(r[10] * 100).toFixed(0)} cm` : ''}</span>` : '';
    el.innerHTML = `${NAMES[k]}<span class="role">${role}</span>${tele}`;
    el.style.left = p.x + 'px'; el.style.top = p.y + 'px'; el.style.opacity = chipOn * (c === 1 ? fxOn('chipsReturn') : 1);
  } }
  // light vehicle chip
  { const show = fxOn('detect') > 0 && det > 0; const p0 = project(lv[0], lv[1], lv[5] + 7.5); const p = { x: p0.x - 230, y: p0.y - 40, ok: p0.ok };
    lvChip.style.opacity = show && p.ok ? 1 : 0; if (show) { lvChip.className = 'chip ' + (det === 2 ? 'hold' : 'lead'); lvChip.innerHTML = `LIGHT VEHICLE<span class="role">${det === 2 ? 'IN PATH' : 'tracked'} · ${detR.toFixed(0)} m</span>`; lvChip.style.left = p.x + 'px'; lvChip.style.top = p.y + 'px'; } }
  // V2V links
  svg.innerHTML = '';
  const v2v = fxOn('v2v');
  if (v2v > 0.01) { const cs = convoyAt(t, 0); const hold = cs.state === 'HOLD' || cs.state === 'CLEAR';
    const pts = [0, 1, 2, 3].map(k => { const r = truckAt(t, 0, k); return project(r[0] + 6.0 * Math.cos(r[2]), r[1] + 6.0 * Math.sin(r[2]), r[15] + 6.0); });
    for (let k = 1; k < 4; k++) { const a = pts[k - 1], b = pts[k]; if (!a.ok || !b.ok) continue;
      const l = document.createElementNS(NS, 'line'); Object.entries({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: hold ? '#ff4d4d' : '#19d3c5', 'stroke-width': 3, 'stroke-dasharray': '10 9', 'stroke-dashoffset': -tv * 60, opacity: 0.85 * v2v }).forEach(([kk, vv]) => l.setAttribute(kk, vv)); svg.appendChild(l);
      for (let q = 0; q < 2; q++) { const u = ((tv * 1.4 + q * 0.5 + k * 0.17) % 1); const c = document.createElementNS(NS, 'circle'); Object.entries({ cx: lerp(a.x, b.x, u), cy: lerp(a.y, b.y, u), r: 5, fill: hold ? '#ff4d4d' : '#9ffff6', opacity: v2v }).forEach(([kk, vv]) => c.setAttribute(kk, vv)); svg.appendChild(c); } }
    for (const p of pts) if (p.ok) { const c = document.createElementNS(NS, 'circle'); Object.entries({ cx: p.x, cy: p.y, r: 7, fill: 'none', stroke: '#fff', 'stroke-width': 2, opacity: 0.8 * v2v }).forEach(([kk, vv]) => c.setAttribute(kk, vv)); svg.appendChild(c); }
  }
  // text layers
  for (const id of Object.keys(STORY.layers)) { const el = document.getElementById('L_' + id); const f = shot.layers && shot.layers[id]; el.style.opacity = f ? f(tau, dur) : 0; }
  chartBox.style.opacity = fxOn('chart'); if (fxOn('chart') > 0) drawChart(t, EV.detect);
  logEl.style.opacity = fxOn('log'); if (fxOn('log') > 0) drawLog(t);
  const sp = shot.speedBadge ? shot.speedBadge(tau, dur) : ''; const sb = document.getElementById('speed'); sb.textContent = sp; sb.style.opacity = sp ? 1 : 0;
  const ck = document.getElementById('clock'); ck.textContent = `T+${t.toFixed(1)} s`; ck.style.opacity = fxOn('clock');
  document.getElementById('sim').style.opacity = fxOn('simBadge') || 0.85;
  document.getElementById('fade').style.opacity = shot.fade ? shot.fade(tau, dur) : 0;
  document.getElementById('dim').style.opacity = shot.dim ? shot.dim(tau, dur) : 0;
  const lb = shot.letterbox ? shot.letterbox(tau, dur) : 0; document.getElementById('lbt').style.height = lb + 'px'; document.getElementById('lbb').style.height = lb + 'px';
  lastShot = shot;
  return t;
};
await document.fonts.ready;
window.READY = true;
