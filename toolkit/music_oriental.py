"""Original oriental-style music for Benchmark reels (royalty-free, synthesized).
make_music(path, dur, seed, maqam='hijaz'|'kurd', bpm=100..115, rhythm='maqsum'|'malfuf', riser_at=sec)"""
import wave
import numpy as np
SR = 44100
MAQAM = {  # semitone offsets from tonic (12-TET approximations)
    "hijaz": [0, 1, 4, 5, 7, 8, 10, 12],
    "kurd":  [0, 1, 3, 5, 7, 8, 10, 12],
}
RHY = {  # 8 eighth-note slots per bar: D=dum, T=tak, .=rest
    "maqsum": "DT.TD.T.",
    "malfuf": "D..T..T.",
}

def ks_pluck(freq, dur, rng, bright=0.5, decay=0.995):
    """Plucked-string tone (qanun/oud-like): bright harmonics, fast attack, per-harmonic decay."""
    t = np.arange(int(dur * SR)) / SR
    sig = np.zeros_like(t)
    for h in range(1, 9):
        sig += (bright ** (h - 1)) / h * np.sin(2*np.pi*freq*h*t*(1 + 0.0004*h*h) + rng.uniform(0, 6.28)) * np.exp(-t * (4 + 2.2*h))
    att = np.minimum(1, t / 0.002)
    return sig * att + 0.15 * rng.standard_normal(len(t)) * np.exp(-t * 120)

def make_music(path, dur, seed=0, maqam="hijaz", bpm=108, rhythm="maqsum", tonic=293.66, riser_at=None):
    n = int(SR * dur); mix = np.zeros(n); rng = np.random.default_rng(seed)
    eighth = 60 / bpm / 2; bar = eighth * 8
    def add(sig, st, g=1.0):
        s = int(st * SR)
        if s >= n or s < 0: return
        e = min(n, s + len(sig)); mix[s:e] += g * sig[:e - s]
    x = lambda L: np.arange(int(L * SR)) / SR
    # darbuka dum (deep) and tak (sharp)
    t = x(0.45); dum = np.sin(2*np.pi*np.cumsum(95*np.exp(-t*6)+60)/SR) * np.exp(-t*7)
    t2 = x(0.12); tak = (rng.standard_normal(len(t2))*0.5 + np.sin(2*np.pi*1800*t2)*0.6) * np.exp(-t2*45)
    t3 = x(0.18); jingle = np.diff(np.concatenate([[0], rng.standard_normal(len(t3))])) * np.exp(-t3*22)
    pat = RHY[rhythm]
    k = 0; st = 0.0
    while st < dur:
        c = pat[k % 8]
        if c == "D": add(dum, st, 0.95)
        elif c == "T": add(tak, st, 0.45)
        if k % 2 == 1: add(jingle, st, 0.18)      # riq/daf on the off-beats (halves)
        if k % 8 == 4 and rng.random() < 0.5: add(tak, st + eighth/2, 0.25)  # ornament
        k += 1; st += eighth
    sc = MAQAM[maqam]
    note = lambda deg, oct=0: tonic * 2 ** ((sc[deg % 7] + 12 * (deg // 7) + 12 * oct) / 12)
    # bass on each bar: tonic / 4th / 5th movement
    bassdeg = [0, 0, 3, 4]
    for b, st in enumerate(np.arange(0, dur, bar)):
        f = note(bassdeg[b % 4], -2); t = x(bar * 0.9)
        sig = (np.sin(2*np.pi*f*t) + 0.3*np.sin(2*np.pi*2*f*t)) * np.exp(-t*1.6) * np.minimum(1, t/0.01)
        add(sig, st, 0.32); add(sig[: int(SR*bar*0.4)], st + eighth*5, 0.2)
    # short repeating motif (2 bars), varied by seed
    motifs = [[0,1,2,1,0,None,4,3, 2,1,2,3,4,None,2,0],
              [4,3,2,1,2,None,0,1, 2,3,4,5,4,3,2,None],
              [0,2,1,3,2,4,3,None, 4,5,4,3,2,1,0,None],
              [2,1,0,1,2,3,4,None, 3,2,1,2,1,0,None,0]]
    mot = motifs[seed % 4]
    cache = {}
    i = 0; st = bar * 1  # melody enters after 1 bar intro
    while st < dur - 1.2:
        d = mot[i % 16]
        if d is not None:
            f = note(d, 0)
            if f not in cache:
                p = ks_pluck(f, 0.6, rng, 0.5, 0.995); cache[f] = p / (np.max(np.abs(p)) + 1e-9)
            add(cache[f], st, 0.30)
            if i % 4 == 0:  # octave doubling for qanun shimmer
                f2 = note(d, 1)
                if f2 not in cache:
                    p = ks_pluck(f2, 0.4, rng, 0.5, 0.993); cache[f2] = p / (np.max(np.abs(p)) + 1e-9)
                add(cache[f2], st + 0.012, 0.12)
        i += 1; st += eighth
    if riser_at is not None:
        t = x(min(2.0, dur - riser_at)); L = len(t)
        sweep = np.sin(2*np.pi*np.cumsum(200 + 1200*(t/t[-1])**2)/SR) * (t/t[-1])**2
        noise = np.convolve(rng.standard_normal(L), np.ones(20)/20, "same") * (t/t[-1])**2
        add(0.22*sweep + 0.35*noise, riser_at)
    fade = np.ones(n); fo = int(1.2*SR); fade[-fo:] = np.linspace(1, 0, fo)
    fi = int(0.05*SR); fade[:fi] *= np.linspace(0, 1, fi)
    mix = np.tanh(mix * 0.9) * fade
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.85
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())
    return path
