"""Builds a responsive, theme-aware web version of the memo (for sharing as a page) from memo.html.

Usage: python venture/memo/build_web.py OUT_DIR   -> OUT_DIR/index.html (+ copies img/*.jpg); put promo_web.mp4 and
poster.jpg next to it. The page body is the memo's own markup, so the two never drift apart.
"""
import re
import shutil
import sys
from pathlib import Path

here = Path(__file__).parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / 'img').mkdir(exist_ok=True)
for f in (here / 'img').glob('*.jpg'):
    shutil.copy(f, out / 'img' / f.name)

src = (here / 'memo.html').read_text()
body = src[src.index('<body>') + 6: src.index('</body>')]
body = re.sub(r'<!-- =+ PAGE \d -->', '', body)
body = body.replace('<section class="page">', '<section>').replace('<div class="cols">', '<div class="flow">')
header_end = body.index('</header>') + len('</header>')
video = """
<figure class="film">
  <video controls playsinline preload="metadata" poster="poster.jpg" src="promo_web.mp4"></video>
  <figcaption>92-second promo, rendered from our convoy simulation: Python vehicle dynamics, V2V, control and perception logic, drawn with Three.js. Simulation, not field footage. The original score is synthesized; press play for sound.</figcaption>
</figure>"""
body = body[:header_end] + video + body[header_end:]

css = """
<title>Leadhaul Pre-Seed Memo</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap">
<style>
/* Layout: one readable column (memo prose ~70ch), wide tables scroll in place, video first. */
:root {
  --bg: #f6f7f5; --surface: #ffffff; --ink: #141a20; --mute: #56606b; --rule: #d9dee2;
  --teal: #0b8f86; --teal-bg: #e4f3f1; --red: #b3261e; --pill: #eceff1;
  --font-body: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; --font-mono: 'JetBrains Mono', ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f1418; --surface: #161c21; --ink: #e8ecef; --mute: #9aa5ae; --rule: #2a333a; --teal: #3fd1c4; --teal-bg: #12302d; --red: #ff7a70; --pill: #222a30; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f1418; --surface: #161c21; --ink: #e8ecef; --mute: #9aa5ae; --rule: #2a333a; --teal: #3fd1c4; --teal-bg: #12302d; --red: #ff7a70; --pill: #222a30; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 16px; line-height: 1.55; }
.wrap { max-width: 860px; margin: 0 auto; padding-block: 28px 56px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-end; gap: 12px; border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 18px; }
.brand { font-size: 30px; font-weight: 800; letter-spacing: 0.12em; }
.brand span { color: var(--teal); }
.brand small { display: block; font-size: 14px; font-weight: 600; letter-spacing: 0.01em; color: var(--mute); margin-top: 2px; }
.meta { font-size: 13px; color: var(--mute); line-height: 1.5; }
.meta b { color: var(--ink); }
.film { margin: 0 0 22px 0; }
.film video { width: 100%; max-width: 100%; aspect-ratio: 16 / 9; display: block; border-radius: 10px; background: #000; }
.film figcaption { font-size: 13px; color: var(--mute); margin-top: 8px; }
.tldr { background: var(--teal-bg); border-left: 4px solid var(--teal); padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 8px; }
.tldr b { color: var(--teal); }
.flow { min-width: 0; }
h2 { font-size: 21px; font-weight: 800; margin: 30px 0 8px 0; text-wrap: balance; }
h2 .n { color: var(--teal); margin-right: 8px; font-variant-numeric: tabular-nums; }
p { margin: 0 0 12px 0; max-width: 70ch; }
ul { margin: 0 0 12px 0; padding-left: 20px; max-width: 70ch; }
li { margin-bottom: 6px; }
.tbl { overflow-x: auto; margin: 6px 0 16px 0; }
table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.4; font-variant-numeric: tabular-nums; min-width: 520px; }
th { text-align: left; font-weight: 700; border-bottom: 1.5px solid var(--ink); padding: 6px 8px; }
td { border-bottom: 1px solid var(--rule); padding: 6px 8px; vertical-align: top; }
td.r, th.r { text-align: right; white-space: nowrap; }
tr.hl td { background: var(--teal-bg); font-weight: 600; }
tr.bad td { color: var(--red); }
.pill { display: inline-block; font-size: 11px; font-weight: 700; padding: 1px 6px; border-radius: 4px; background: var(--pill); color: var(--mute); letter-spacing: 0.04em; vertical-align: 1px; }
.mono { font-family: var(--font-mono); font-size: 0.9em; }
.ask { border: 1.5px solid var(--ink); border-radius: 10px; padding: 14px 18px; margin: 10px 0 16px 0; background: var(--surface); }
.ask .big { font-size: 22px; font-weight: 800; }
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0 6px 0; }
.strip figure { margin: 0; min-width: 0; }
.strip img { width: 100%; display: block; border-radius: 6px; }
.strip figcaption { font-size: 13px; color: var(--mute); margin-top: 6px; }
.strip figcaption b { color: var(--teal); }
footer { margin-top: 30px; border-top: 1px solid var(--rule); padding-top: 12px; font-size: 13px; color: var(--mute); }
a { color: var(--teal); }
</style>
"""
body = re.sub(r'(<table[^>]*>.*?</table>)', r'<div class="tbl">\1</div>', body, flags=re.S)
(out / 'index.html').write_text(css + '<div class="wrap">' + body + '</div>\n')
print('wrote', out / 'index.html')
