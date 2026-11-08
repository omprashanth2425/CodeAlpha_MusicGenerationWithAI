import io
import os
import random
import sys

import mido
import torch

from music_common import (
    HIGH, HOLD, LOW, MAJOR_TOK, MINOR_TOK, PITCH_OFFSET, REST, STEPS_PER_BEAT,
    MelodyLSTM, pitch_to_token, token_to_pitch,
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "melody_lstm.pt")

STEPS_PER_BAR = 16
PHRASE_STEPS = 32      # one phrase = 2 bars
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
INSTRUMENTS = {"piano": 0, "music box": 10, "guitar": 24, "violin": 40, "flute": 73}
MOODS = {
    "happy": {"mode": MAJOR_TOK, "tempo": 118, "temp": 0.80, "style": "oompah"},
    "calm":  {"mode": MAJOR_TOK, "tempo": 76,  "temp": 0.65, "style": "arp"},
    "sad":   {"mode": MINOR_TOK, "tempo": 68,  "temp": 0.70, "style": "pad"},
    "dark":  {"mode": MINOR_TOK, "tempo": 96,  "temp": 0.85, "style": "drive"},
}
CHORD_PROGRAM = {"oompah": 0, "arp": 46, "pad": 89, "drive": 50}
BASS_PROGRAM = {"oompah": 32, "arp": 32, "pad": 32, "drive": 38}

# ---- Music theory (everything is composed in C major / A minor, then transposed)
SCALE = {0, 2, 4, 5, 7, 9, 11}
HALF_CADENCE = {"major": ([7, 2], [4, 9]), "minor": ([4, 0], [2, 7])}

MAJOR_CHORDS = {"I": (0, [0, 4, 7]), "ii": (2, [2, 5, 9]), "iii": (4, [4, 7, 11]),
                "IV": (5, [5, 9, 0]), "V": (7, [7, 11, 2]), "vi": (9, [9, 0, 4])}
MINOR_CHORDS = {"i": (9, [9, 0, 4]), "iv": (2, [2, 5, 9]), "V": (4, [4, 8, 11]),
                "III": (0, [0, 4, 7]), "VI": (5, [5, 9, 0]), "VII": (7, [7, 11, 2])}
MAJOR_NEXT = {"I": ("IV", "V", "vi", "ii"), "ii": ("V",), "iii": ("vi", "IV"),
              "IV": ("V", "I", "ii"), "V": ("I", "vi"), "vi": ("IV", "ii", "V")}
MINOR_NEXT = {"i": ("iv", "VI", "VII", "V"), "iv": ("V", "i", "VII"), "V": ("i", "VI"),
              "III": ("VI", "iv", "VII"), "VI": ("VII", "iv", "III", "V"), "VII": ("i", "III")}

_PITCHES = torch.tensor([token_to_pitch(t) for t in range(PITCH_OFFSET, MAJOR_TOK)],
                        dtype=torch.float32)
_SCALE_PEN_MAJOR = torch.tensor([0.0 if int(p) % 12 in SCALE else 2.5 for p in _PITCHES])
_SCALE_PEN_MINOR = torch.tensor([0.0 if int(p) % 12 in SCALE else (1.0 if int(p) % 12 == 8 else 2.5)
                                 for p in _PITCHES])

_MODEL = None
_MODEL_TRIED = False


def get_model():
    global _MODEL, _MODEL_TRIED
    if _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location="cpu")
            model = MelodyLSTM(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _MODEL = model
            print("[generator] LSTM model loaded.")
        else:
            print("[generator] No trained model found, using fallback generator.")
    except Exception as e:
        print("[generator] Could not load model:", e)
    return _MODEL


# ---- Melody samplers ----------------------------------------------------
class Sampler:
    """Asks the LSTM for notes, but steers it like a composer would."""

    def __init__(self, model, mode_tok, temperature, seed, is_major):
        self.model = model
        self.g = torch.Generator().manual_seed(int(seed))
        self.temp = max(float(temperature), 0.05)
        self.scale_pen = _SCALE_PEN_MAJOR if is_major else _SCALE_PEN_MINOR
        self.state = None
        self.prev = mode_tok
        self.pos = 0
        self.last_pitch = None
        self.sounding = False

    def _next_logits(self):
        with torch.no_grad():
            logits, self.state = self.model(torch.tensor([[self.prev]]), self.state)
        return logits[0, -1].clone()

    def _track(self, t):
        if PITCH_OFFSET <= t < MAJOR_TOK:
            self.last_pitch = token_to_pitch(t)
            self.sounding = True
        elif t == REST:
            self.sounding = False
        self.prev = t
        self.pos += 1

    def feed(self, tokens):
        """Show the model music we copied, so it continues from the right place."""
        for t in tokens:
            self._next_logits()
            self._track(int(t))

    def sample(self, n):
        out = []
        for _ in range(n):
            logits = self._next_logits()
            logits[MAJOR_TOK:] = -float("inf")
            body = logits[PITCH_OFFSET:MAJOR_TOK]
            body -= self.scale_pen                                   # stay in the scale
            if self.last_pitch is not None:
                leap = (_PITCHES - self.last_pitch).abs()
                body -= 0.06 * leap + 0.35 * torch.clamp(leap - 7, min=0)   # prefer smooth motion
            logits[REST] -= 1.5                                      # fewer random gaps
            if self.pos % 2 == 1:                                    # new notes only on the eighth-note grid
                logits[PITCH_OFFSET:MAJOR_TOK] = -float("inf")
            if self.pos % PHRASE_STEPS == 0:                         # every phrase starts with a clear note
                logits[REST] = -float("inf")
                logits[HOLD] = -float("inf")
            if not self.sounding:
                logits[HOLD] = -float("inf")

            logits = logits / self.temp
            probs = torch.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            keep = (torch.cumsum(sp, 0) - sp) < 0.92                 # nucleus sampling
            sp = sp * keep
            sp = sp / sp.sum()
            choice = int(si[torch.multinomial(sp, 1, generator=self.g)].item())
            self._track(choice)
            out.append(choice)
        return out


class FallbackSampler:
    """Used only if the trained model is missing: a simple scale-based melody."""

    def __init__(self, rng, is_major):
        self.rng = rng
        self.scale = [p for p in range(55, 80) if p % 12 in SCALE]
        self.idx = self.scale.index(60 if is_major else 69)

    def feed(self, tokens):
        pass

    def sample(self, n):
        out = []
        while len(out) < n:
            self.idx = max(0, min(len(self.scale) - 1, self.idx + self.rng.choice([-2, -1, -1, 0, 1, 1, 2])))
            dur = self.rng.choice([2, 2, 2, 4, 4, 8])
            out.append(pitch_to_token(self.scale[self.idx]))
            out.extend([HOLD] * (dur - 1))
        return out[:n]


# ---- Song structure -----------------------------------------------------
def _plan_form(bars):
    """Phrases of 2 bars. Pattern A B A C | D E D F, so ideas come back and repeat."""
    sizes = [PHRASE_STEPS] * (bars // 2) + ([STEPS_PER_BAR] if bars % 2 else [])
    groups = [["A", "B", "A", "C"], ["D", "E", "D", "F"]]
    return [(groups[(i // 4) % 2][i % 4], size) for i, size in enumerate(sizes)]


def _cadence(tokens, pcs, hold):
    """Ends a phrase on a held note from `pcs`, like a musical full stop or comma."""
    tokens = list(tokens)
    n = len(tokens)
    hold = min(hold, n)
    last = next((token_to_pitch(t) for t in reversed(tokens[:n - hold])
                 if PITCH_OFFSET <= t < MAJOR_TOK), 66)
    cands = [p for p in range(LOW, HIGH + 1) if p % 12 in pcs]
    p = min(cands, key=lambda c: abs(c - last))
    tokens[n - hold] = pitch_to_token(p)
    for i in range(n - hold + 1, n):
        tokens[i] = HOLD
    return tokens


def _compose(sampler, is_major, bars):
    plan = _plan_form(bars)
    half = HALF_CADENCE["major" if is_major else "minor"]
    tonic_pc = 0 if is_major else 9
    made, tokens = {}, []
    for idx, (letter, size) in enumerate(plan):
        if letter in made:                         # an idea coming back
            phrase = made[letter][:size]
            sampler.feed(phrase)
        else:                                      # a new idea from the AI
            phrase = sampler.sample(size)
            made[letter] = list(phrase)
        role = idx % 4
        if idx == len(plan) - 1 or role == 3:
            phrase = _cadence(phrase, [tonic_pc], 8)    # full stop on the home note
        elif role == 1:
            phrase = _cadence(phrase, half[1], 4)       # a comma
        else:
            phrase = _cadence(phrase, half[0], 4)       # a question
        tokens += phrase
    return tokens


def _tokens_to_notes(tokens):
    notes, cur = [], None
    for i, t in enumerate(tokens):
        if PITCH_OFFSET <= t < MAJOR_TOK:
            cur = [i, 1, token_to_pitch(t)]
            notes.append(cur)
        elif t == HOLD and cur is not None:
            cur[1] += 1
        else:
            cur = None
    return notes


def _penalty(tokens, bars):
    """Scores how musical a melody is. Lower is better."""
    notes = _tokens_to_notes(tokens)
    if len(notes) < 4:
        return 99.0
    pitches = [n[2] for n in notes]
    dens = len(notes) / bars
    pen = 0.0
    if dens < 3:
        pen += 3 - dens
    if dens > 12:
        pen += (dens - 12) * 0.5
    pen += max(0.0, tokens.count(REST) / len(tokens) - 0.15) * 8
    big = sum(1 for a, b in zip(pitches, pitches[1:]) if abs(a - b) > 9)
    pen += 3.0 * big / len(notes)
    span = max(pitches) - min(pitches)
    if span > 24:
        pen += 1.0
    if span < 5:
        pen += 1.5
    return pen


# ---- Arrangement --------------------------------------------------------
def _stack(pcs, base):
    out, cur = [], base - 1
    for pc in pcs:
        p = cur + 1
        while p % 12 != pc:
            p += 1
        out.append(p)
        cur = p
    return out


def _voice_chord(pcs, prev):
    """Picks the chord inversion that moves the least from the previous chord."""
    best = None
    for i in range(3):
        v = _stack(pcs[i:] + pcs[:i], 52)
        cost = abs(sum(v) / 3 - 60) if prev is None else sum(abs(a - b) for a, b in zip(v, prev))
        if best is None or cost < best[0]:
            best = (cost, v)
    return best[1]


def _accompaniment(notes, bars, is_major, style, rng):
    lib = MAJOR_CHORDS if is_major else MINOR_CHORDS
    nxt = MAJOR_NEXT if is_major else MINOR_NEXT
    names = list(lib)
    tonic = names[0]

    # 1) choose a chord for every bar
    progression, prev = [], None
    for b in range(bars):
        s0, s1 = b * STEPS_PER_BAR, (b + 1) * STEPS_PER_BAR
        if b == 0 or b == bars - 1:
            name = tonic
        else:
            weight = {}
            for start, dur, pitch in notes:
                overlap = min(start + dur, s1) - max(start, s0)
                if overlap > 0:
                    w = overlap + (2 if start in (s0, s0 + 8) else 0)
                    weight[pitch % 12] = weight.get(pitch % 12, 0) + w
            best = None
            for cand in names:
                score = sum(weight.get(pc, 0) for pc in lib[cand][1]) / 4.0
                if prev is not None:
                    if cand in nxt.get(prev, ()):
                        score += 1.5
                    if cand == prev:
                        score -= 1.0
                if b == bars - 2 and cand == "V":
                    score += 4.0                      # lead into the final chord
                if best is None or score > best[0]:
                    best = (score, cand)
            name = best[1]
        progression.append(name)
        prev = name

    # 2) turn the chords into a playing pattern
    def vel(base, spread):
        return base + rng.randint(-spread, spread)

    chords, bass, prev_v = [], [], None
    for b, name in enumerate(progression):
        s0 = b * STEPS_PER_BAR
        root, pcs = lib[name]
        v = _voice_chord(pcs, prev_v)
        prev_v = v
        bp = 36 + root
        if b == bars - 1:                              # final chord rings out
            for p in v:
                chords.append([s0, 16, p, 54])
            bass.append([s0, 16, bp, 72])
            break
        if style == "oompah":
            bass += [[s0, 6, bp, vel(80, 3)], [s0 + 8, 6, bp + 7, vel(68, 3)]]
            for st in (4, 12):
                for p in v:
                    chords.append([s0 + st, 3, p, vel(58, 4)])
        elif style == "arp":
            for k, i in enumerate([0, 1, 2, 1, 0, 1, 2, 1]):
                chords.append([s0 + 2 * k, 3, v[i], vel(62 if k % 4 == 0 else 48, 4)])
            bass.append([s0, 14, bp, vel(70, 3)])
        elif style == "pad":
            for p in v:
                chords.append([s0, 16, p, vel(46, 3)])
            bass.append([s0, 14, bp, vel(66, 3)])
        else:  # drive
            for k in range(8):
                bass.append([s0 + 2 * k, 2, bp, vel(82 if k % 4 == 0 else 62, 3)])
            for st in (0, 8):
                for p in v:
                    chords.append([s0 + st, 8, p, vel(54, 4)])
    return chords, bass


def _drums(bars, style, rng):
    out = []
    for b in range(bars):
        s0 = b * STEPS_PER_BAR
        if b == bars - 1:
            out += [[s0, 4, 36, 100], [s0, 8, 49, 80]]     # kick + crash to finish
            break
        if style == "oompah":
            for st in (0, 8):
                out.append([s0 + st, 2, 36, rng.randint(96, 104)])
            for st in (4, 12):
                out.append([s0 + st, 2, 38, rng.randint(78, 88)])
            for st in range(0, 16, 2):
                out.append([s0 + st, 1, 42, 80 if st % 4 == 0 else 62])
        elif style == "drive":
            for st in (0, 4, 8, 12):
                out.append([s0 + st, 2, 36, rng.randint(98, 106)])
            for st in (4, 12):
                out.append([s0 + st, 2, 38, rng.randint(80, 90)])
            for st in (2, 6, 10, 14):
                out.append([s0 + st, 1, 42, rng.randint(64, 74)])
        else:  # soft styles
            out.append([s0, 2, 36, rng.randint(78, 88)])
            out.append([s0 + 8, 2, 37, rng.randint(60, 70)])
            for st in (4, 12):
                out.append([s0 + st, 1, 42, rng.randint(46, 56)])
    return out


# ---- MIDI file ----------------------------------------------------------
def _build_midi(melody, chords, bass, drums, tempo, mel_prog, chord_prog, bass_prog):
    tpb = 480
    step_ticks = tpb // STEPS_PER_BEAT
    mid = mido.MidiFile(ticks_per_beat=tpb)

    def add_track(notes, channel, prog, name, first=False):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if first:
            track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
            track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        if prog is not None:
            track.append(mido.Message("program_change", channel=channel, program=prog, time=0))
        events = []
        for start, dur, pitch, velocity in notes:
            events.append((start * step_ticks, 1, pitch, velocity))
            events.append(((start + dur) * step_ticks, 0, pitch, 0))
        events.sort(key=lambda e: (e[0], e[1]))      # note-offs before note-ons
        last = 0
        for tick, is_on, pitch, velocity in events:
            track.append(mido.Message(
                "note_on" if is_on else "note_off", channel=channel, note=pitch,
                velocity=velocity if is_on else 0, time=tick - last))
            last = tick

    add_track(melody, 0, mel_prog, "Melody", first=True)
    add_track(chords, 1, chord_prog, "Chords")
    add_track(bass, 2, bass_prog, "Bass")
    if drums:
        add_track(drums, 9, None, "Drums")

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _clamp(x, lo, hi):
    return max(lo, min(hi, int(x)))


# ---- Public function ----------------------------------------------------
def generate(mood="happy", key="C", tempo=None, bars=8, temperature=None,
             seed=None, instrument="piano", drums=False):
    mood = mood if mood in MOODS else "happy"
    key = key if key in KEYS else "C"
    instrument = instrument if instrument in INSTRUMENTS else "piano"
    cfg = MOODS[mood]
    style = cfg["style"]
    is_major = cfg["mode"] == MAJOR_TOK

    bars = max(2, min(32, int(bars)))
    tempo = _clamp(tempo if tempo else cfg["tempo"], 50, 200)
    temperature = max(0.3, min(1.5, float(temperature) if temperature else cfg["temp"]))
    seed = random.randint(0, 999999) if seed is None else int(seed)

    # compose the melody: try a few candidates, keep the most musical one
    tokens, engine, best = None, "fallback", None
    model = get_model()
    if model is not None:
        for attempt in range(4):
            try:
                sampler = Sampler(model, cfg["mode"], temperature, seed + attempt * 7919, is_major)
                cand = _compose(sampler, is_major, bars)
            except Exception as e:
                print("[generator] Model sampling failed:", e)
                break
            pen = _penalty(cand, bars)
            if best is None or pen < best[0]:
                best = (pen, cand)
            if pen < 1.2:
                break
    if best is not None and best[0] < 50:
        tokens, engine = best[1], "lstm"
    else:
        tokens = _compose(FallbackSampler(random.Random(seed), is_major), is_major, bars)

    # transpose into the requested key
    root = KEYS.index(key)
    shift = root if is_major else (root - 9) % 12
    if shift > 6:
        shift -= 12

    rng = random.Random(seed + 12345)
    base_notes = _tokens_to_notes(tokens)
    chord_notes, bass_notes = _accompaniment(base_notes, bars, is_major, style, rng)
    drum_notes = _drums(bars, style, rng) if drums else []

    melody = []
    for s, d, p in base_notes:
        accent = 10 if s % 8 == 0 else (4 if s % 4 == 0 else 0)
        melody.append([s, d, _clamp(p + shift, 0, 127), _clamp(84 + accent + rng.randint(-5, 5), 1, 127)])
    chords = [[s, d, _clamp(p + shift, 0, 127), _clamp(v, 1, 127)] for s, d, p, v in chord_notes]
    bass = [[s, d, _clamp(p + shift, 0, 127), _clamp(v, 1, 127)] for s, d, p, v in bass_notes]

    steps = bars * STEPS_PER_BAR
    midi_bytes = _build_midi(melody, chords, bass, drum_notes, tempo,
                             INSTRUMENTS[instrument], CHORD_PROGRAM[style], BASS_PROGRAM[style])
    return {
        "seed": seed, "mood": mood, "key": key,
        "mode": "major" if is_major else "minor",
        "tempo": tempo, "bars": bars, "temperature": temperature,
        "instrument": instrument, "engine": engine, "style": style,
        "steps_per_beat": STEPS_PER_BEAT, "total_steps": steps,
        "duration_sec": round(steps / STEPS_PER_BEAT * 60 / tempo, 1),
        "melody": melody, "chords": chords, "bass": bass, "drums": drum_notes,
        "midi": midi_bytes,
    }


if __name__ == "__main__":
    mood = sys.argv[1] if len(sys.argv) > 1 else "happy"
    key = sys.argv[2] if len(sys.argv) > 2 else "C"
    bars = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with_drums = len(sys.argv) > 4 and sys.argv[4] == "drums"
    result = generate(mood=mood, key=key, bars=bars, drums=with_drums)
    out_dir = os.path.join(BASE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test.mid")
    with open(path, "wb") as f:
        f.write(result["midi"])
    print(f"Engine: {result['engine']} | {result['mood']} in {result['key']} {result['mode']} "
          f"| {result['tempo']} BPM | {result['bars']} bars | {result['duration_sec']} sec "
          f"| {len(result['melody'])} melody notes | seed {result['seed']}")
    print("Saved:", path)