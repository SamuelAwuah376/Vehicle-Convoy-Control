// Shot list, camera moves and on-screen copy for the Leadhaul promo (92 s at 30 fps).
// Each shot maps video time (tau, seconds into the shot) to sim time, a camera, HUD effects and text layers.
export const FPS = 30;

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const L3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
// opacity: fade in over [a, a+fi], hold, fade out over [b-fo, b]
const win = (a, b, fi = 0.6, fo = 0.6) => tau => Math.min(ease((tau - a) / fi), 1 - ease((tau - (b - fo)) / fo));
// piecewise-linear sim time from (tau, sim) knots
const knots = pts => tau => { for (let i = 1; i < pts.length; i++) if (tau <= pts[i][0]) return lerp(pts[i - 1][1], pts[i][1], (tau - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0])); return pts[pts.length - 1][1]; };

function chase(t, ctx, c, kBack, kAim, back, side, up, fov) {
  const r = ctx.truckAt(t, c, kBack), a = ctx.truckAt(t, c, kAim), b = ctx.truckAt(t, c, Math.max(0, kAim - 1));
  const h = ctx.headingSmooth(t, c, kBack, 2.0);
  const pos = [r[0] - back * Math.cos(h) - side * Math.sin(h), r[1] - back * Math.sin(h) + side * Math.cos(h), r[15] + up];
  const target = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[15] + b[15]) / 2 + 2.5];
  return { pos, target, fov, focus: [(r[0] + a[0]) / 2, (r[1] + a[1]) / 2, a[15]] };
}
function centroid(t, ctx, c = 0) { let x = 0, y = 0, z = 0; for (let k = 0; k < 4; k++) { const r = ctx.truckAt(t, c, k); x += r[0]; y += r[1]; z += r[15]; } return [x / 4, y / 4, z / 4]; }

const shots = [
  { id: 'title', t0: 0, t1: 6, sim: tau => 1 + 0.6 * tau,
    cam: (tau, d) => { const u = ease(tau / d); return { pos: L3([-340, -300, 215], [-285, -250, 182], u), target: L3([-25, 15, -24], [-30, 20, -24], u), fov: 38, focus: [-20, 20, -30] }; },
    fade: tau => 1 - ease((tau - 0.6) / 1.6), layers: { title: win(1.4, 6.0, 0.9, 0.7) }, fx: { simBadge: 0 } },

  { id: 'problem', t0: 6, t1: 15, sim: tau => 4.6 + 2.2 * tau,
    cam: (tau, d, t, ctx) => { const u = ease(tau / d); const az = (158 + 48 * u) * Math.PI / 180; const c = [-55, 10, -30]; const L = ctx.truckAt(t, 0, 0);
      return { pos: [c[0] + 180 * Math.cos(az), c[1] + 180 * Math.sin(az), 70 - 18 * u], target: [lerp(c[0], L[0], 0.55), lerp(c[1], L[1], 0.55), -27], fov: 36, focus: [L[0], L[1], -30] }; },
    layers: { problem: win(0.5, 9.0, 0.7, 0.6) }, speedBadge: () => '2×' },

  { id: 'solution', t0: 15, t1: 27, sim: tau => 26 + 3.0 * tau,
    cam: (tau, d, t, ctx) => chase(t, ctx, 0, 3, 1, 36, -11, 17, 44),
    layers: { solution: win(0.8, 11.6, 0.7, 0.6) }, fx: { chips: tau => ease((tau - 1.2) / 0.8), v2v: tau => ease((tau - 3.5) / 1.0) }, speedBadge: () => '3×' },

  { id: 'tech', t0: 27, t1: 41, sim: tau => 93 + 2.0 * tau,
    cam: (tau, d, t, ctx) => { const c = centroid(t, ctx); const u = ease(tau / d); const tg = [c[0] + 38, c[1] + 4, c[2]]; return { pos: [tg[0] + lerp(30, -5, u), tg[1] - lerp(100, 88, u), tg[2] + lerp(118, 104, u)], target: tg, fov: 44, focus: c }; },
    layers: { tech: win(0.6, 14, 0.6, 0.6) },
    fx: { chips: 1, tele: 1, v2v: 1, crumbs: tau => ease(tau / 1.0), corridors: tau => ease((tau - 5.0) / 1.0), chipsReturn: 1, detect: 1, chipMask: [1180, 120, 1900, 700] }, speedBadge: () => '2×' },

  { id: 'safety', t0: 41, t1: 57, sim: knots([[0, 120], [4, 126], [11, 133], [16, 143]]),
    cam: (tau, d) => { const u = ease(tau / d); return { pos: L3([292, 44, 33], [300, 50, 30], u), target: L3([356, 127, 0], [362, 127, 0], u), fov: 40, focus: [360, 125, 0] }; },
    layers: { safety: win(0.5, 6.5, 0.6, 0.6) },
    fx: { chips: 1, tele: 1, v2v: 1, corridors: 1, detect: 1, chart: win(1.0, 16, 0.6, 0.4), log: win(4.0, 16, 0.4, 0.4), clock: 1, chipMask: [1200, 140, 1900, 320] },
    speedBadge: tau => (tau < 4 ? '1.5×' : tau < 11 ? '1×' : '2×') },

  { id: 'dump', t0: 57, t1: 69, sim: tau => 176 + 4.5 * tau,
    cam: (tau, d) => { const u = ease(tau / d); return { pos: L3([606, 182, 21], [612, 190, 24], u), target: L3([646, 230, 3], [649, 232, 3], u), fov: 44, focus: [640, 230, 0] }; },
    layers: { dump: win(0.6, 11.4, 0.6, 0.6) }, fx: { chips: 1 }, speedBadge: () => '4.5×' },

  { id: 'business', t0: 69, t1: 83, sim: tau => 230 + 2.0 * tau,
    cam: (tau, d) => { const u = ease(tau / d); return { pos: L3([700, 110, 105], [500, -270, 390], u), target: L3([520, 190, -2], [235, 60, -12], u), fov: 40, focus: L3([600, 220, 0], [250, 60, 0], u) }; },
    layers: { stats: win(1.0, 14, 0.8, 0.6) } },

  { id: 'end', t0: 83, t1: 92, sim: tau => 258 + 0.2 * tau,
    cam: (tau) => { const u = ease(tau / 9); return { pos: L3([500, -270, 390], [470, -300, 410], u), target: L3([235, 60, -12], [230, 60, -12], u), fov: 40, focus: [250, 60, 0] }; },
    dim: tau => 0.8 * ease(tau / 1.2), fade: tau => ease((tau - 8.2) / 0.8), layers: { end: win(0.8, 8.9, 0.9, 0.7) }, fx: { simBadge: 0 } },
];

export const TOTAL_FRAMES = Math.round(shots[shots.length - 1].t1 * FPS);

export const STORY = {
  shots,
  layers: {
    title: () => `<div class="card"><div class="logo">LEAD<span>HAUL</span></div><div class="tag">One operator. The whole haul.</div>
      <div class="desc">Human-led autonomous haul convoys for mines and quarries</div></div>`,
    problem: () => `<div class="lower"><div class="kicker">The problem</div>
      <div class="line">Haul trucks run the same loop all day.<br>Every one of them still needs a driver.</div>
      <div class="sub">77% of US contractors with openings struggle to hire heavy-equipment operators.<br>An empty seat means an idle truck and lost tonnage.</div>
      <div class="src">Source: AGC 2025 Workforce Survey</div></div>`,
    solution: () => `<div class="lower"><div class="kicker">The idea</div>
      <div class="line">One operator leads.<br>Up to three driverless trucks follow.</div>
      <div class="sub">A retrofit kit turns trucks a site already owns into followers.<br>No site survey. No remote control room.</div></div>`,
    tech: m => { const rms = Math.max(...['AUTO 1', 'AUTO 2', 'AUTO 3'].map(k => m.loaded[k].lat_rms_cm));
      return `<div class="panel"><h3>How it works</h3>
      <div class="item"><div class="t">Follows the operator's exact line</div><div class="d">Each follower tracks the leader's path from V2V breadcrumbs, so a haul road that moves needs no re-mapping.</div><div class="m">${rms.toFixed(1)} cm RMS path error · simulated</div></div>
      <div class="item"><div class="t">Cooperative spacing</div><div class="d">Speed and braking are shared 20 times a second; gaps grow with speed and stopping distance.</div><div class="m">≈50 m gap at 35 km/h · 40–60 ms V2V, 2% loss</div></div>
      <div class="item"><div class="t">Every follower sees for itself</div><div class="d">Radar and lidar check the corridor each truck will sweep, not just the truck in front.</div><div class="m">80 m corridor · 2.5 s look-ahead</div></div></div>`; },
    safety: () => `<div class="lower" style="max-width:1000px"><div class="kicker" style="color:#ff6b6b">Safety</div>
      <div class="line">Any truck can stop<br>the whole convoy.</div>
      <div class="sub">A site pickup runs the crossing. AUTO 1 sees it enter its path and the convoy holds over V2V. The operator resumes when it is clear.</div></div>`,
    dump: () => `<div class="lower"><div class="kicker">At the dump</div>
      <div class="line">Each truck spots and dumps itself.</div>
      <div class="sub">The operator assigns dump points from the cab tablet. Followers peel off, tip, and form up again.</div></div>`,
    stats: () => `<div class="stats-title"><small>The business</small>Turn one operator into a fleet.</div>
      <div class="stats">
        <div class="stat"><div class="n">1<span>→</span>4</div><div class="l">trucks moved by each operator on convoy routes</div></div>
        <div class="stat"><div class="n">75<span>%</span></div><div class="l">fewer driver-hours per ton on those routes</div></div>
        <div class="stat"><div class="n">Any<span>OEM</span></div><div class="l">retrofit kit for the trucks a site already owns</div></div>
        <div class="stat"><div class="n"><span>$</span>/ton</div><div class="l">priced per ton hauled, so sites pay for production</div></div>
      </div>
      <div class="note">Rendered from our convoy simulation (vehicle dynamics, V2V, control and perception logic in Python; Three.js). Business figures are design targets, not field results.</div>`,
    end: () => `<div class="card"><div class="logo">LEAD<span>HAUL</span></div><div class="tag">One operator. The whole haul.</div>
      <div class="desc">Raising a pre-seed round · looking for design-partner quarries and mines</div>
      <div class="foot">Leadhaul is a working name · simulation, not field footage</div></div>`,
  },
};
