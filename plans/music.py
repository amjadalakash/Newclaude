"""Original oriental-style music for Benchmark reels (royalty-free, synthesized).
make_music(path, dur, maqam='hijaz', bpm=108, seed=0, riser_at=None, rhythm='maqsum')"""
import wave
import numpy as np
SR = 44100
MAQAMAT = {  # semitone offsets from tonic (D)
    "hijaz": [0, 1, 4, 5, 7, 8, 10, 12, 13, 16],
    "kurd":  [0, 1, 3, 5, 7, 8, 10, 12, 13, 15],
    "nahawand": [0, 2, 3, 5, 7, 8, 11, 12, 14, 15],
}
RHYTHMS = {  # 8 eighth-note slots per bar: D=dum, T=tek, '-'=rest
    "maqsum": "DT-TD-T-",
    "malfuf": "D--T--T-",
    "saidi":  "DT-DD-T-",
}
def _ks(freq, dur, rng, bright=0.996):
    """Karplus-Strong pluck (qanun/oud-like)."""
    n = int(SR * dur); p = max(2, int(SR / freq))
    buf = rng.uniform(-1, 1, p); out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = bright * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out

def make_music(path, dur, maqam="hijaz", bpm=108, seed=0, riser_at=None, rhythm="maqsum", tonic=146.83):
    rng = np.random.default_rng(seed)
    n = int(SR * dur); mix = np.zeros(n)
    def add(sig, st, g=1.0):
        s = int(st * SR)
        if s >= n or s < 0: return
        e = min(n, s + len(sig)); mix[s:e] += g * sig[:e - s]
    eighth = 60 / bpm / 2; bar = eighth * 8
    scale = MAQAMAT[maqam]; pat = RHYTHMS[rhythm]
    def note(deg, octv=0):
        return tonic * 2 ** ((scale[deg % len(scale)] + 12 * octv) / 12)
    # percussion
    def dum():
        L = int(0.4 * SR); x = np.arange(L) / SR
        f = 70 + 60 * np.exp(-x * 30)
        return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-x * 9) + 0.08 * rng.standard_normal(L) * np.exp(-x * 60)
    def tek():
        L = int(0.12 * SR); x = np.arange(L) / SR
        nz = rng.standard_normal(L); nz = nz - np.convolve(nz, np.ones(6) / 6, "same")
        return 0.55 * nz * np.exp(-x * 45) + 0.25 * np.sin(2 * np.pi * 1800 * x) * np.exp(-x * 70)
    def riq():
        L = int(0.18 * SR); x = np.arange(L) / SR
        nz = rng.standard_normal(L); nz = np.diff(np.concatenate([[0], nz]))
        jingle = sum(np.sin(2 * np.pi * f * x) for f in (5200, 6900, 8300)) / 3
        return 0.12 * (nz * 0.7 + jingle * 0.5) * np.exp(-x * 22)
    t = 0.0; k = 0
    while t < dur:
        c = pat[k % 8]
        if c == "D": add(dum(), t)
        elif c == "T": add(tek(), t)
        if k % 2 == 1 or c == "-": add(riq(), t)   # riq on off-beats
        if k % 2 == 0: add(riq(), t, 0.5)
        t += eighth; k += 1
    # bass on dums following phrase roots
    roots = [0, 0, 4, 3]  # degree roots per bar
    for b, st in enumerate(np.arange(0, dur, bar)):
        r = roots[b % 4]
        for slot, c in enumerate(pat):
            if c == "D":
                L = int(eighth * 1.8 * SR); x = np.arange(L) / SR
                f = note(r, -2)
                sig = (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * 2 * f * x)) * np.exp(-x * 4) * np.minimum(1, x / 0.005)
                add(0.42 * sig, st + slot * eighth)
    # melody motif (2 bars, 16 eighths); -1 = rest. vary by seed
    motifs = [
        [7, -1, 6, 5, 4, -1, 5, 4, 3, 1, 3, 4, 0, -1, -1, -1],
        [4, 5, 7, -1, 7, 8, 7, 5, 4, -1, 3, 1, 0, -1, 1, 3],
        [0, 1, 3, 4, 5, -1, 4, 5, 7, 5, 4, 3, 1, -1, 0, -1],
        [7, 7, 5, 4, 5, 4, 3, 1, 3, -1, 1, 0, 1, 3, 4, -1],
    ]
    mot = motifs[seed % len(motifs)]
    for b2, st in enumerate(np.arange(bar * 1, dur - bar, bar * 2)):  # enter after 1 bar
        octv = 1 if b2 % 2 == 1 else 0
        for i, d in enumerate(mot):
            if d < 0: continue
            nl = eighth * (2 if (i + 1 < 16 and mot[i + 1] == -1) else 1.2)
            sig = _ks(note(d, octv), nl + 0.25, rng)
            sig *= np.minimum(1, np.arange(len(sig)) / (0.003 * SR))
            add(0.32 * sig, st + i * eighth)
            if i % 4 == 0:  # tremolo-like double (qanun)
                add(0.12 * _ks(note(d, octv), 0.15, rng), st + i * eighth + eighth / 2)
    # drone pad (soft)
    x = np.arange(n) / SR
    drone = 0.05 * (np.sin(2 * np.pi * tonic / 2 * x) + 0.5 * np.sin(2 * np.pi * tonic * 1.5 / 2 * x))
    mix += drone * np.minimum(1, x / 2.0)
    # riser before finale
    if riser_at is not None:
        L = int(2.0 * SR); xx = np.arange(L) / SR
        nz = rng.standard_normal(L); nz = nz - np.convolve(nz, np.ones(20) / 20, "same")
        sweep = np.sin(2 * np.pi * np.cumsum(300 + 1400 * (xx / 2) ** 2) / SR)
        add((0.18 * nz + 0.15 * sweep) * (xx / 2) ** 2, riser_at)
        add(1.1 * dum(), riser_at + 2.0)   # impact
    # fades & normalize (no clipping)
    fade = np.ones(n); fi = int(0.05 * SR); fo = int(1.2 * SR)
    fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo)
    mix = mix * fade
    mix = mix / np.percentile(np.abs(mix), 99.7) * 0.7
    mix = np.tanh(mix)  # gentle soft-limit on rare transients only
    mix = mix / np.max(np.abs(mix)) * 0.85
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())
    return float(np.max(np.abs(mix))), n / SR
