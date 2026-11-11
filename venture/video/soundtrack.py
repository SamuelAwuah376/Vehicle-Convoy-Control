"""Original synthesized soundtrack for the promo (no samples, no licensed music): writes soundtrack.wav.

Pads (A minor - F - C - G), a sub bass, an eighth-note pulse, transition swells, and a two-tone alert at the
moment AUTO 1 detects the light vehicle; the pulse drops out for the safety beat and returns on resume.
Cue times follow story.js (shot boundaries) and the sim events (detect at sim 129.25 s -> video 46.25 s).
"""
import wave
from pathlib import Path

import numpy as np

SR = 44100
DUR = 92.0
BPM = 96.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(3)

SHOTS = [0, 6, 15, 27, 41, 57, 69, 83, 92]
T_DETECT, T_RESUME = 46.25, 55.55


def note(n):            # MIDI -> Hz
    return 440.0 * 2 ** ((n - 69) / 12)


def env_adsr(length, a, r):
    e = np.ones(length)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) if na else e[:na]
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def lowpass(x, alpha):
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):          # one-pole filter
        acc += alpha * (x[i] - acc)
        y[i] = acc
    return y


def section_gain(times, pts):
    return np.interp(times, [p[0] for p in pts], [p[1] for p in pts])


out = np.zeros((N, 2))

# --- pads: 4 chords, 5 s each, cycling; detuned triangle-ish voices
chords = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 55, 60, 64], [55, 59, 62, 67]]   # Am F C G
chord_len = 5.0
for ci, start in enumerate(np.arange(0.0, DUR, chord_len)):
    ch = chords[ci % 4]
    i0, i1 = int(start * SR), min(N, int((start + chord_len + 1.5) * SR))
    tt = t[i0:i1] - start
    e = env_adsr(i1 - i0, 1.6, 2.0)
    for k, n in enumerate(ch):
        for det, pan in ((-0.07, 0.3), (0.07, 0.7)):
            f = note(n) * (1 + det / 100)
            w = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * tt + k))      # triangle
            w += 0.25 * np.sin(2 * np.pi * 2 * f * tt)
            out[i0:i1, 0] += 0.018 * e * w * (1 - pan)
            out[i0:i1, 1] += 0.018 * e * w * pan
pad_gain = section_gain(t, [(0, 0.0), (2.5, 1.0), (41, 1.0), (43, 0.55), (55, 0.55), (57, 1.0), (83, 1.0), (92, 0.0)])
out *= pad_gain[:, None]

# --- sub bass on the beat (from the idea section), root of each chord
beat = 60.0 / BPM
roots = [45, 41, 36, 43]
bass = np.zeros(N)
for b in np.arange(15.0, 83.0, beat):
    if T_DETECT - 0.5 < b < T_RESUME:
        continue
    ci = int(b // chord_len) % 4
    i0, i1 = int(b * SR), min(N, int((b + beat * 0.95) * SR))
    tt = t[i0:i1] - b
    bass[i0:i1] += 0.09 * np.sin(2 * np.pi * note(roots[ci]) * tt) * np.exp(-tt * 3.0) * (1 - np.exp(-tt * 200))
out += bass[:, None]

# --- pulse: eighth-note plucks arpeggiating the chord (tech, dump, business sections)
pulse = np.zeros((N, 2))
for j, b in enumerate(np.arange(27.0, 83.0, beat / 2)):
    if T_DETECT - 0.3 < b < T_RESUME + 0.2 or 41.0 <= b < T_DETECT - 0.3:
        continue
    ci = int(b // chord_len) % 4
    n = chords[ci][j % 4] + 12
    i0, i1 = int(b * SR), min(N, int((b + 0.5) * SR))
    tt = t[i0:i1] - b
    w = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 9.0) * (1 - np.exp(-tt * 400))
    pan = 0.35 if j % 2 else 0.65
    pulse[i0:i1, 0] += 0.045 * w * (1 - pan)
    pulse[i0:i1, 1] += 0.045 * w * pan
out += pulse

# --- low drone under the safety beat
i0, i1 = int(41.0 * SR), int(57.0 * SR)
tt = t[i0:i1] - 41.0
dr = 0.05 * np.sin(2 * np.pi * note(33) * tt) * env_adsr(i1 - i0, 1.5, 2.0) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * tt))
out[i0:i1] += dr[:, None]

# --- alert: two-tone ping at detection, a softer resolve on resume
for (tc, notes, amp) in ((T_DETECT, (88, 84), 0.11), (T_DETECT + 0.45, (88, 84), 0.08), (T_RESUME, (76, 81), 0.07)):
    for k, n in enumerate(notes):
        b = tc + k * 0.14
        j0, j1 = int(b * SR), min(N, int((b + 0.8) * SR))
        tt = t[j0:j1] - b
        w = np.sin(2 * np.pi * note(n) * tt) * np.exp(-tt * 6.0) * (1 - np.exp(-tt * 300))
        out[j0:j1] += amp * w[:, None]

# --- swells into each shot change (filtered noise)
noise = rng.standard_normal(N)
sw = np.zeros(N)
for s in SHOTS[1:-1]:
    a, b = s - 1.2, s + 0.25
    j0, j1 = int(a * SR), int(b * SR)
    shape = np.concatenate([np.linspace(0, 1, int(1.2 * SR)) ** 2, np.linspace(1, 0, j1 - j0 - int(1.2 * SR))])
    sw[j0:j1] += shape * noise[j0:j1]
sw = lowpass(sw, 0.035) * 0.12
out += np.column_stack([sw * 0.9, sw * 1.1])

# --- master: gentle stereo room (short delays), fade, normalise
room = np.zeros_like(out)
for d, g in ((0.031, 0.25), (0.047, 0.2), (0.089, 0.14), (0.137, 0.1)):
    k = int(d * SR)
    room[k:, 0] += g * out[:-k, 1]
    room[k:, 1] += g * out[:-k, 0]
out = out + room
fade = np.clip(np.minimum(t / 1.5, (DUR - t) / 2.5), 0, 1)
out *= fade[:, None]
out /= np.max(np.abs(out)) / 0.82
pcm = (out * 32767).astype('<i2')
path = Path(__file__).with_name('soundtrack.wav')
with wave.open(str(path), 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f'wrote {path} ({DUR:.0f} s)')
