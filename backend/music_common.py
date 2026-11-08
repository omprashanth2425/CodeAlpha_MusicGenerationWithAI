import numpy as np
import mido
import torch.nn as nn

# ---- Token vocabulary -------------------------------------------------
LOW, HIGH = 48, 84                      # pitch range (C3 to C6)
REST, HOLD = 0, 1                       # 0 = silence, 1 = keep holding the note
PITCH_OFFSET = 2                        # tokens 2..38 are pitches
MAJOR_TOK = PITCH_OFFSET + (HIGH - LOW + 1)   # 39 = "major mood" start token
MINOR_TOK = MAJOR_TOK + 1                     # 40 = "minor mood" start token
VOCAB = MINOR_TOK + 1                         # 41 tokens in total
STEPS_PER_BEAT = 4                      # 16th-note grid

# Krumhansl-Schmuckler key profiles (used to detect a tune's key)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_to_token(p):
    return p - LOW + PITCH_OFFSET


def token_to_pitch(t):
    return t - PITCH_OFFSET + LOW


# ---- The neural network ----------------------------------------------
class MelodyLSTM(nn.Module):
    def __init__(self, vocab=VOCAB, embed=64, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, state=None):
        out, state = self.lstm(self.embed(x), state)
        return self.fc(out), state


# ---- MIDI reading -----------------------------------------------------
def _read_note_groups(mid):
    """Returns {(track, channel): [(start_tick, end_tick, pitch), ...]}"""
    groups = {}
    for ti, track in enumerate(mid.tracks):
        t = 0
        active = {}
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.channel == 9:      # channel 10 = drums, skip
                    continue
                active[(msg.channel, msg.note)] = t
            elif msg.type in ("note_off", "note_on"):
                if msg.channel == 9:
                    continue
                key = (msg.channel, msg.note)
                if key in active:
                    start = active.pop(key)
                    groups.setdefault((ti, msg.channel), []).append((start, t, msg.note))
    return groups


def detect_key(pitches, durations):
    hist = np.zeros(12)
    for p, d in zip(pitches, durations):
        hist[p % 12] += d
    if hist.std() == 0:
        return None
    best = None
    for tonic in range(12):
        for mode, prof in ((0, MAJOR_PROFILE), (1, MINOR_PROFILE)):
            c = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if best is None or c > best[0]:
                best = (c, tonic, mode)
    return best[1], best[2]


def midi_to_tokens(path):
    """Turns a MIDI file into (token_array, mode_token), or None if unusable."""
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    tpb = mid.ticks_per_beat
    if not tpb:
        return None

    groups = {k: v for k, v in _read_note_groups(mid).items() if len(v) >= 8}
    if not groups:
        return None
    # melody = the track/channel with the highest average pitch
    melody = max(groups.values(), key=lambda n: np.mean([x[2] for x in n]))

    step_ticks = tpb / STEPS_PER_BEAT
    onsets = {}
    for s, e, p in melody:
        qs = int(round(s / step_ticks))
        qe = max(int(round(e / step_ticks)), qs + 1)
        if qs not in onsets or p > onsets[qs][1]:   # keep the highest note
            onsets[qs] = (qe, p)

    starts = sorted(onsets)
    if len(starts) < 8:
        return None
    pitches = [onsets[s][1] for s in starts]
    durs = [onsets[s][0] - s for s in starts]

    key = detect_key(pitches, durs)
    if key is None:
        return None
    tonic, mode = key
    target = 0 if mode == 0 else 9           # C for major, A for minor
    shift = (target - tonic) % 12
    if shift > 6:
        shift -= 12

    ends = []
    for i, s in enumerate(starts):
        qe = onsets[s][0]
        if i + 1 < len(starts):
            qe = min(qe, starts[i + 1])      # monophonic: cut at next note
        ends.append(qe)

    first = starts[0]
    total = ends[-1] - first
    if total <= 0 or total > 20000:
        return None

    seq = np.full(total, REST, dtype=np.int64)
    for s, qe, p in zip(starts, ends, pitches):
        p += shift
        while p < LOW:
            p += 12
        while p > HIGH:
            p -= 12
        seq[s - first] = pitch_to_token(p)
        seq[s - first + 1: qe - first] = HOLD
    return seq, (MAJOR_TOK if mode == 0 else MINOR_TOK)