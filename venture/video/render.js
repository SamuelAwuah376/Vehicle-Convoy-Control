// Renders the promo frame by frame in headless Chromium (SwiftShader WebGL) and encodes with ffmpeg.
//   node render.js                      all frames -> frames/, then promo.mp4 (with soundtrack.wav if present)
//   node render.js --frames 0,450,900   stills only -> preview/
//   node render.js --range 0 600        a slice (for parallel workers); --no-encode skips ffmpeg
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path'), { execFileSync } = require('child_process');
const ROOT = __dirname;
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const has = k => args.includes(k);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };
function serve() {
  return new Promise(res => { const srv = http.createServer((req, rsp) => {
    const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { rsp.writeHead(404); rsp.end(); return; }
    rsp.writeHead(200, { 'Content-Type': TYPES[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(rsp);
  }); srv.listen(0, '127.0.0.1', () => res(srv)); });
}
(async () => {
  const srv = await serve(); const port = srv.address().port;
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') console.log('[page]', m.text()); });
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${port}/index.html${opt('--query') ? '?' + opt('--query') : ''}`);
  await page.waitForFunction(() => window.READY === true, null, { timeout: 900000 });
  const total = await page.evaluate(() => window.TOTAL_FRAMES);
  let list;
  if (opt('--frames')) list = opt('--frames').split(',').map(Number);
  else { const a = Number(opt('--range', 0)); const b = args.includes('--range') ? Number(args[args.indexOf('--range') + 2]) : total; list = []; for (let i = a; i < Math.min(b, total); i++) list.push(i); }
  const still = !!opt('--frames'); const outDir = path.join(ROOT, opt('--out', still ? 'preview' : 'frames')); fs.mkdirSync(outDir, { recursive: true });
  const t0 = Date.now();
  for (const [n, i] of list.entries()) {
    const out = path.join(outDir, `f_${String(i).padStart(5, '0')}.${still ? 'png' : 'jpg'}`);
    if (!still && fs.existsSync(out) && !has('--force')) continue;
    const simT = await page.evaluate(i => window.renderFrame(i), i);
    await page.screenshot({ path: out, type: still ? 'png' : 'jpeg', quality: still ? undefined : 94 });
    if (n % 25 === 0 || still) console.log(`frame ${i} (sim ${simT.toFixed(1)} s) ${((Date.now() - t0) / 1000 / (n + 1)).toFixed(2)} s/frame`);
  }
  await browser.close(); srv.close();
  if (!still && !has('--no-encode')) {
    const audio = fs.existsSync(path.join(ROOT, 'soundtrack.wav'));
    const ff = ['-y', '-framerate', '30', '-i', path.join(outDir, 'f_%05d.jpg')];
    if (audio) ff.push('-i', path.join(ROOT, 'soundtrack.wav'));
    ff.push('-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart');
    if (audio) ff.push('-c:a', 'aac', '-b:a', '160k', '-shortest');
    ff.push(path.join(ROOT, 'promo.mp4'));
    execFileSync('ffmpeg', ff, { stdio: 'inherit' });
  }
})();
