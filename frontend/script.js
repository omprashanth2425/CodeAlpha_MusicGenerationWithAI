"use strict";

const $ = (id) => document.getElementById(id);

const KEYS_DEFAULT = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const INSTR_DEFAULT = ["piano", "music box", "guitar", "violin", "flute"];
const HKEY = "melodia.history";
const COLORS = {
  grid: "rgba(255,255,255,0.07)", note: "rgba(139,92,246,0.5)", played: "rgba(167,139,250,0.95)",
  active: "#22d3ee", head: "#ffffff", text: "rgba(255,255,255,0.4)",
};

const state = {
  piece: null, playing: false, paused: false, busy: false,
  ctx: null, chain: null, t0: 0, raf: 0, lastPos: -1,
  creativityTouched: false,
};

// ---------- Small helpers ----------
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

let toastTimer = 0;
function toast(msg, isError) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "show" + (isError ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ""; }, 4500);
}

function setBadge(kind, text) {
  const b = $("badge");
  b.className = "badge " + kind;
  b.textContent = text;
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
  } catch (e) {
    throw new Error("Cannot reach the Melodia server. Make sure it is running (python app.py).");
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : `Request failed (${res.status}). Check your settings.`);
  }
  return data;
}

// ---------- Controls ----------
function fillSelect(id, items, selected) {
  const el = $(id);
  el.innerHTML = "";
  items.forEach((v) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = cap(v);
    if (v === selected) o.selected = true;
    el.appendChild(o);
  });
}

function currentMood() {
  return document.querySelector(".mood.active").dataset.mood;
}

function setMood(m) {
  document.querySelectorAll(".mood").forEach((b) => b.classList.toggle("active", b.dataset.mood === m));
}

function creativityLabel(v) {
  return v < 0.7 ? "Predictable" : v < 1.1 ? "Balanced" : "Adventurous";
}

function readParams() {
  const p = {
    mood: currentMood(),
    key: $("key").value,
    instrument: $("instrument").value,
    bars: parseInt($("bars").value, 10),
    drums: $("drums").checked,
  };
  if (!$("tempoAuto").checked) p.tempo = parseInt($("tempo").value, 10);
  if (state.creativityTouched) p.temperature = parseFloat($("creativity").value);
  const seed = $("seed").value.trim();
  if (seed !== "") p.seed = parseInt(seed, 10);
  return p;
}

function applyParams(p) {
  setMood(p.mood);
  $("key").value = p.key;
  $("instrument").value = p.instrument;
  $("bars").value = String(p.bars);
  $("drums").checked = !!p.drums;
  $("tempoAuto").checked = false;
  $("tempo").disabled = false;
  $("tempo").value = p.tempo;
  $("tempoOut").textContent = p.tempo + " BPM";
  state.creativityTouched = true;
  $("creativity").value = p.temperature;
  $("creativityOut").textContent = creativityLabel(p.temperature);
  $("seed").value = p.seed;
}

function setBusy(b) {
  state.busy = b;
  const btn = $("generate");
  btn.disabled = b;
  btn.classList.toggle("loading", b);
  $("genLabel").textContent = b ? "Composing" : "Generate music";
}

// ---------- Generate ----------
async function generate(params) {
  if (state.busy) return;
  try { ensureAudio(); state.ctx.resume(); } catch (e) { /* audio unavailable */ }
  setBusy(true);
  stop();
  try {
    const piece = await api("/api/generate", params || readParams());
    setPiece(piece);
    addHistory({
      mood: piece.mood, key: piece.key, instrument: piece.instrument, bars: piece.bars,
      tempo: piece.tempo, temperature: piece.temperature, seed: piece.seed,
      drums: !!(piece.drums && piece.drums.length),
    });
    await play();
  } catch (e) {
    toast(e.message || "Something went wrong. Please try again.", true);
  } finally {
    setBusy(false);
  }
}

function pieceName() {
  const p = state.piece;
  return `Melodia_${p.mood}_${p.key}_${p.seed}`;
}

function setPiece(p) {
  state.piece = p;
  $("title").textContent = `${cap(p.mood)} in ${p.key} ${p.mode}`;

  const chips = $("chips");
  chips.innerHTML = "";
  const labels = [`${p.tempo} BPM`, `${p.bars} bars`, cap(p.instrument)];
  if (p.drums && p.drums.length) labels.push("Drums");
  labels.push(p.engine === "lstm" ? "AI: LSTM" : "AI: backup");
  labels.forEach((t) => {
    const c = document.createElement("span");
    c.className = "chip"; c.textContent = t;
    chips.appendChild(c);
  });
  const seedChip = document.createElement("button");
  seedChip.type = "button";
  seedChip.className = "chip";
  seedChip.textContent = `Seed ${p.seed} (use again)`;
  seedChip.onclick = () => { $("seed").value = p.seed; toast("Seed copied into the form."); };
  chips.appendChild(seedChip);

  ["playBtn", "stopBtn", "midiBtn", "wavBtn"].forEach((id) => { $(id).disabled = false; });
  setTime(0);
  updateTransport();
  drawRoll(-1);
}

// ---------- Piano-roll visualizer ----------
function drawRoll(pos) {
  state.lastPos = pos;
  const cv = $("roll");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const p = state.piece;
  if (!p) {
    g.fillStyle = COLORS.text;
    g.font = "500 15px system-ui, sans-serif";
    g.textAlign = "center";
    g.fillText("Choose a mood and press Generate", w / 2, h / 2);
    return;
  }

  const total = p.total_steps;
  g.strokeStyle = COLORS.grid;
  g.lineWidth = 1;
  for (let s = 0; s <= total; s += 16) {
    const x = Math.round((s / total) * w) + 0.5;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }

  const pitches = p.melody.map((n) => n[2]);
  let lo = Math.min(...pitches) - 2, hi = Math.max(...pitches) + 2;
  if (hi - lo < 12) { lo = Math.floor((hi + lo) / 2 - 6); hi = lo + 12; }
  const rows = hi - lo + 1, rh = h / rows;
  const stepSec = 60 / (p.tempo * p.steps_per_beat);
  const cur = pos >= 0 ? pos / stepSec : -1;

  p.melody.forEach(([s, d, pitch]) => {
    const x = (s / total) * w;
    const nw = Math.max((d / total) * w - 1.5, 2);
    const y = h - (pitch - lo + 1) * rh + rh * 0.12;
    const nh = Math.max(rh * 0.76, 3);
    const active = cur >= s && cur < s + d;
    const played = cur >= s + d;
    g.fillStyle = active ? COLORS.active : played ? COLORS.played : COLORS.note;
    if (active) { g.shadowColor = COLORS.active; g.shadowBlur = 14; }
    g.beginPath();
    if (g.roundRect) g.roundRect(x, y, nw, nh, 3); else g.rect(x, y, nw, nh);
    g.fill();
    g.shadowBlur = 0;
  });

  if (cur >= 0) {
    const x = Math.min(cur / total, 1) * w;
    g.strokeStyle = COLORS.head;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke();
  }
}

// =====================================================================
//  SOUND ENGINE (built with the Web Audio API, no libraries)
// =====================================================================
const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
const amp = (v) => 0.25 + 0.75 * v;            // louder when played harder

const noiseCache = new WeakMap();
function getNoise(c) {
  let b = noiseCache.get(c);
  if (!b) {
    b = c.createBuffer(1, c.sampleRate, c.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    noiseCache.set(c, b);
  }
  return b;
}

// A room: random noise that fades away makes a convincing reverb tail
function makeImpulse(c, secs, decay) {
  const rate = c.sampleRate, len = Math.floor(rate * secs);
  const buf = c.createBuffer(2, len, rate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    let lp = 0;
    for (let i = 0; i < len; i++) {
      lp += 0.35 * ((Math.random() * 2 - 1) - lp);
      d[i] = lp * Math.pow(1 - i / len, decay);
    }
  }
  return buf;
}

function makeChain(c) {
  const dry = c.createGain();
  const wet = c.createGain();
  const conv = c.createConvolver();
  conv.buffer = makeImpulse(c, 2.4, 2.8);
  const wetOut = c.createGain();
  wetOut.gain.value = 0.55;
  const mix = c.createGain();
  mix.gain.value = 0.9;
  const comp = c.createDynamicsCompressor();
  comp.threshold.value = -16;
  comp.ratio.value = 4;
  comp.attack.value = 0.005;
  comp.release.value = 0.2;
  dry.connect(mix);
  wet.connect(conv); conv.connect(wetOut); wetOut.connect(mix);
  mix.connect(comp); comp.connect(c.destination);
  return { dry, wet, out: comp };
}

// One mixer channel per part (melody, chords, bass, drums)
function makeBus(c, chain, level, pan, send) {
  const g = c.createGain();
  g.gain.value = level;
  let node = g;
  if (c.createStereoPanner) {
    const p = c.createStereoPanner();
    p.pan.value = pan;
    g.connect(p);
    node = p;
  }
  node.connect(chain.dry);
  const s = c.createGain();
  s.gain.value = send;
  node.connect(s);
  s.connect(chain.wet);
  return g;
}

function noiseHit(c, out, t, len, type, freq, q, peak) {
  const s = c.createBufferSource();
  s.buffer = getNoise(c);
  s.loop = true;
  const f = c.createBiquadFilter();
  f.type = type; f.frequency.value = freq; f.Q.value = q;
  const g = c.createGain();
  g.gain.setValueAtTime(peak, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  s.connect(f); f.connect(g); g.connect(out);
  s.start(t, Math.random() * 0.5);
  s.stop(t + len + 0.05);
}

function toneHit(c, out, t, len, f0, f1, type, peak) {
  const o = c.createOscillator();
  o.type = type;
  o.frequency.setValueAtTime(f0, t);
  o.frequency.exponentialRampToValueAtTime(f1, t + len * 0.4);
  const g = c.createGain();
  g.gain.setValueAtTime(peak, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + len);
  o.connect(g); g.connect(out);
  o.start(t);
  o.stop(t + len + 0.05);
}

// ----- Instruments -----
// Piano and bells: a stack of sine waves where the high ones fade faster
const PIANO = [[1, 1.0, 1.4], [2, 0.55, 0.8], [3, 0.30, 0.5], [4, 0.18, 0.35], [5, 0.10, 0.25], [6, 0.06, 0.18]];
const BELL = [[1, 1.0, 0.9], [2.76, 0.4, 0.35], [5.4, 0.18, 0.2], [8.93, 0.07, 0.1]];

function additive(c, out, midi, t, dur, v, table, o) {
  const f = hz(midi);
  const k = Math.pow(2, -(midi - 60) / 30) * (o.decayScale || 1);   // high notes fade faster
  const end = t + Math.max(dur, o.minLen);
  const stopAt = end + o.rel * 3 + 0.1;
  table.forEach(([mult, level, tau]) => {
    const freq = f * mult;
    if (freq > 12000) return;
    const osc = c.createOscillator();
    osc.type = "sine";
    osc.frequency.value = freq;
    osc.detune.value = (Math.random() - 0.5) * 6;
    const g = c.createGain();
    const peak = o.gain * level * (mult === 1 ? 1 : 0.4 + 0.6 * v) * amp(v);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(peak, t + o.a);
    g.gain.setTargetAtTime(0.0001, t + o.a, tau * k);
    g.gain.setTargetAtTime(0.0001, end, o.rel / 3);
    osc.connect(g); g.connect(out);
    osc.start(t); osc.stop(stopAt);
  });
  if (o.click) noiseHit(c, out, t, 0.03, "bandpass", 3000, 0.8, o.click * v);   // hammer
}

// Guitar and harp: Karplus-Strong, a simulated vibrating string
function pluck(c, out, midi, t, dur, v, o) {
  const sr = c.sampleRate, f = hz(midi);
  const N = Math.max(2, Math.round(sr / f - 0.5));
  const secs = Math.min(o.maxLen, Math.max(dur, 0.2) + 0.6);
  const len = Math.floor(sr * secs);
  const buf = c.createBuffer(1, len, sr);
  const d = buf.getChannelData(0);
  const coef = 0.25 + 0.75 * o.bright * (0.5 + 0.5 * v);
  let lp = 0, peak = 0;
  for (let i = 0; i < N; i++) {
    lp += coef * ((Math.random() * 2 - 1) - lp);
    d[i] = lp;
    peak = Math.max(peak, Math.abs(lp));
  }
  if (peak > 0) for (let i = 0; i < N; i++) d[i] /= peak;
  const decay = Math.pow(o.decay, 440 / f);
  for (let i = N; i < len; i++) d[i] = decay * 0.5 * (d[i - N] + d[i - N + 1]);

  const src = c.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = f * (N + 0.5) / sr;
  const g = c.createGain();
  const end = t + Math.max(dur, 0.2);
  const level = o.gain * amp(v);
  g.gain.setValueAtTime(level, t);
  g.gain.setValueAtTime(level, end);
  g.gain.setTargetAtTime(0.0001, end, o.damp);       // the string gets damped
  src.connect(g); g.connect(out);
  src.start(t);
  src.stop(t + secs + 0.1);
}

// Violin: two slightly detuned saw waves, body resonances, and delayed vibrato
function bowed(c, out, midi, t, dur, v, o) {
  const f = hz(midi);
  const end = t + Math.max(dur, 0.18);
  const env = c.createGain();
  env.gain.setValueAtTime(0.0001, t);
  env.gain.linearRampToValueAtTime(o.gain * amp(v), t + o.a);
  env.gain.setTargetAtTime(0.0001, end, o.rel / 3);
  const lp = c.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = Math.min(f * 7, 6000);
  const pk1 = c.createBiquadFilter();
  pk1.type = "peaking"; pk1.frequency.value = 1300; pk1.Q.value = 1.1; pk1.gain.value = 7;
  const pk2 = c.createBiquadFilter();
  pk2.type = "peaking"; pk2.frequency.value = 2900; pk2.Q.value = 1.4; pk2.gain.value = 4;
  lp.connect(pk1); pk1.connect(pk2); pk2.connect(env); env.connect(out);

  const stopAt = end + o.rel * 3 + 0.1;
  const lfo = c.createOscillator();
  lfo.frequency.value = o.vib;
  const lg = c.createGain();
  lg.gain.setValueAtTime(0, t);
  lg.gain.linearRampToValueAtTime(f * 0.006, t + 0.3);
  lfo.connect(lg);
  [-7, 7].forEach((cents) => {
    const osc = c.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.value = f;
    osc.detune.value = cents;
    lg.connect(osc.frequency);
    const og = c.createGain();
    og.gain.value = 0.5;
    osc.connect(og); og.connect(lp);
    osc.start(t); osc.stop(stopAt);
  });
  lfo.start(t); lfo.stop(stopAt);
}

// Flute: a pure tone plus a little breath noise
function breathy(c, out, midi, t, dur, v, o) {
  const f = hz(midi);
  const end = t + Math.max(dur, 0.18);
  const env = c.createGain();
  env.gain.setValueAtTime(0.0001, t);
  env.gain.linearRampToValueAtTime(o.gain * amp(v), t + o.a);
  env.gain.setTargetAtTime(0.0001, end, o.rel / 3);
  env.connect(out);
  const stopAt = end + o.rel * 3 + 0.1;

  const lfo = c.createOscillator();
  lfo.frequency.value = o.vib;
  const lg = c.createGain();
  lg.gain.setValueAtTime(0, t);
  lg.gain.linearRampToValueAtTime(f * 0.004, t + 0.35);
  lfo.connect(lg);
  [[1, 1], [2, 0.12]].forEach(([mult, level]) => {
    const osc = c.createOscillator();
    osc.type = "sine";
    osc.frequency.value = f * mult;
    lg.connect(osc.frequency);
    const og = c.createGain();
    og.gain.value = level;
    osc.connect(og); og.connect(env);
    osc.start(t); osc.stop(stopAt);
  });
  lfo.start(t); lfo.stop(stopAt);

  const n = c.createBufferSource();
  n.buffer = getNoise(c);
  n.loop = true;
  const bp = c.createBiquadFilter();
  bp.type = "bandpass"; bp.frequency.value = Math.min(f * 2, 8000); bp.Q.value = 3;
  const ng = c.createGain();
  ng.gain.value = 0.35;
  n.connect(bp); bp.connect(ng); ng.connect(env);
  n.start(t, Math.random() * 0.5);
  n.stop(stopAt);
}

// Pads, synth stabs and basses: layered oscillators through a filter
function layered(c, out, midi, t, dur, v, o) {
  const f = hz(midi);
  const end = t + Math.max(dur, o.minLen || 0.12);
  const peak = o.gain * amp(v);
  const env = c.createGain();
  env.gain.setValueAtTime(0.0001, t);
  env.gain.linearRampToValueAtTime(peak, t + o.a);
  env.gain.setTargetAtTime(Math.max(peak * o.sus, 0.0001), t + o.a, o.dec / 3);
  env.gain.setTargetAtTime(0.0001, end, o.rel / 3);
  const lp = c.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = o.cutoff; lp.Q.value = o.q || 0.7;
  lp.connect(env); env.connect(out);
  const stopAt = end + o.rel * 3 + 0.1;
  o.wave.forEach(([type, cents]) => {
    const osc = c.createOscillator();
    osc.type = type;
    osc.frequency.value = f;
    osc.detune.value = cents;
    const g = c.createGain();
    g.gain.value = 1 / o.wave.length;
    osc.connect(g); g.connect(lp);
    osc.start(t); osc.stop(stopAt);
  });
}

const INSTRUMENT = {
  "piano":     (c, o, m, t, d, v) => additive(c, o, m, t, d, v, PIANO, { gain: 0.2, a: 0.004, rel: 0.45, minLen: 0.3, click: 0.05 }),
  "music box": (c, o, m, t, d, v) => additive(c, o, m < 84 ? m + 12 : m, t, d, v, BELL, { gain: 0.24, a: 0.002, rel: 0.35, minLen: 0.25 }),
  "guitar":    (c, o, m, t, d, v) => pluck(c, o, m, t, d, v, { maxLen: 2.2, decay: 0.996, bright: 0.7, gain: 0.55, damp: 0.12 }),
  "violin":    (c, o, m, t, d, v) => bowed(c, o, m, t, d, v, { gain: 0.22, a: 0.09, rel: 0.2, vib: 5.3 }),
  "flute":     (c, o, m, t, d, v) => breathy(c, o, m, t, d, v, { gain: 0.3, a: 0.06, rel: 0.15, vib: 5 }),
};

const CHORD_VOICE = {
  oompah: (c, o, m, t, d, v) => additive(c, o, m, t, d, v, PIANO, { gain: 0.11, a: 0.005, rel: 0.3, minLen: 0.2, decayScale: 0.8 }),
  arp:    (c, o, m, t, d, v) => pluck(c, o, m, t, d, v, { maxLen: 2.6, decay: 0.9975, bright: 0.4, gain: 0.4, damp: 0.35 }),
  pad:    (c, o, m, t, d, v) => layered(c, o, m, t, d, v, {
    wave: [["sawtooth", -9], ["sawtooth", 0], ["sawtooth", 9]],
    gain: 0.2, a: 0.35, sus: 1, dec: 0.3, rel: 0.9, cutoff: 1100, minLen: 0.5 }),
  drive:  (c, o, m, t, d, v) => layered(c, o, m, t, d, v, {
    wave: [["square", -5], ["sawtooth", 5]],
    gain: 0.14, a: 0.01, sus: 0.45, dec: 0.5, rel: 0.12, cutoff: 1800, q: 2 }),
};

const BASS_VOICE = {
  base:  (c, o, m, t, d, v) => layered(c, o, m, t, d, v, {
    wave: [["sine", 0], ["triangle", 0]],
    gain: 0.5, a: 0.01, sus: 0.6, dec: 0.4, rel: 0.12, cutoff: 600 }),
  drive: (c, o, m, t, d, v) => layered(c, o, m, t, d, v, {
    wave: [["sawtooth", 0], ["square", -8]],
    gain: 0.4, a: 0.005, sus: 0.4, dec: 0.2, rel: 0.06, cutoff: 420, q: 1.5 }),
};

function drum(c, out, note, t, vel) {
  const v = Math.max(0.1, vel / 127);
  if (note === 36) {
    toneHit(c, out, t, 0.38, 150, 45, "sine", 0.95 * v);                              // kick
  } else if (note === 38) {
    noiseHit(c, out, t, 0.2, "bandpass", 1900, 0.8, 0.55 * v);                        // snare
    toneHit(c, out, t, 0.12, 220, 160, "triangle", 0.35 * v);
  } else if (note === 37) {
    noiseHit(c, out, t, 0.05, "highpass", 2500, 0.7, 0.4 * v);                        // side stick
    toneHit(c, out, t, 0.05, 520, 420, "triangle", 0.3 * v);
  } else if (note === 42) {
    noiseHit(c, out, t, 0.06, "highpass", 7500, 0.7, 0.28 * v);                       // hi-hat
  } else if (note === 49) {
    noiseHit(c, out, t, 1.6, "highpass", 4500, 0.7, 0.4 * v);                         // crash
  }
}

function schedule(c, chain, piece, t0) {
  const stepSec = 60 / (piece.tempo * piece.steps_per_beat);
  const style = piece.style || "oompah";
  const melBus = makeBus(c, chain, 1.0, -0.1, 0.30);
  const chordBus = makeBus(c, chain, 0.9, 0.1, 0.40);
  const bassBus = makeBus(c, chain, 0.9, 0.0, 0.04);
  const drumBus = makeBus(c, chain, 0.6, 0.0, 0.12);

  const melodyPlayer = INSTRUMENT[piece.instrument] || INSTRUMENT.piano;
  const chordPlayer = CHORD_VOICE[style] || CHORD_VOICE.oompah;
  const bassPlayer = style === "drive" ? BASS_VOICE.drive : BASS_VOICE.base;

  const addNotes = (notes, player, bus) => (notes || []).forEach(([s, d, pitch, vel]) =>
    player(c, bus, pitch, t0 + s * stepSec, d * stepSec, (vel || 90) / 127));
  addNotes(piece.melody, melodyPlayer, melBus);
  addNotes(piece.chords, chordPlayer, chordBus);
  addNotes(piece.bass, bassPlayer, bassBus);
  (piece.drums || []).forEach(([s, d, note, vel]) => drum(c, drumBus, note, t0 + s * stepSec, vel || 90));
}

function ensureAudio() {
  if (state.ctx) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  state.ctx = new AC();
}

// ---------- Transport ----------
function setTime(pos) {
  const dur = state.piece ? state.piece.duration_sec : 0;
  $("time").textContent = `${fmt(Math.max(0, Math.min(pos, dur)))} / ${fmt(dur)}`;
}

function updateTransport() {
  const b = $("playBtn");
  b.textContent = state.playing ? "⏸" : "▶";
  b.setAttribute("aria-label", state.playing ? "Pause" : "Play");
}

function loop() {
  cancelAnimationFrame(state.raf);
  const tick = () => {
    if (!state.playing) return;
    const pos = state.ctx.currentTime - state.t0;
    if (pos >= state.piece.duration_sec + 0.3) { stop(); return; }
    drawRoll(Math.max(0, pos));
    setTime(pos);
    state.raf = requestAnimationFrame(tick);
  };
  tick();
}

async function play() {
  if (!state.piece) return;
  ensureAudio();
  if (state.paused) {
    await state.ctx.resume();
    state.paused = false; state.playing = true;
    updateTransport(); loop();
    return;
  }
  stop(true);
  await state.ctx.resume();
  state.chain = makeChain(state.ctx);
  state.t0 = state.ctx.currentTime + 0.2;
  schedule(state.ctx, state.chain, state.piece, state.t0);
  state.playing = true; state.paused = false;
  updateTransport(); loop();
}

function pause() {
  if (!state.playing) return;
  state.ctx.suspend();
  state.playing = false; state.paused = true;
  updateTransport();
}

function stop(silent) {
  cancelAnimationFrame(state.raf);
  if (state.chain) { try { state.chain.out.disconnect(); } catch (e) { /* already gone */ } state.chain = null; }
  if (state.ctx && state.ctx.state === "suspended") state.ctx.resume();
  state.playing = false; state.paused = false;
  if (!silent) {
    updateTransport();
    drawRoll(-1);
    setTime(0);
  }
}

// ---------- Downloads ----------
function downloadMidi() {
  const bin = atob(state.piece.midi_base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  download(new Blob([bytes], { type: "audio/midi" }), pieceName() + ".mid");
}

function encodeWav(buf) {
  const ch = buf.numberOfChannels, n = buf.length, sr = buf.sampleRate;
  const chans = [];
  let peak = 0;
  for (let k = 0; k < ch; k++) {
    const d = buf.getChannelData(k);
    chans.push(d);
    for (let i = 0; i < n; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; }
  }
  const gain = peak > 0 ? Math.min(0.92 / peak, 4) : 1;      // normalize the volume
  const ab = new ArrayBuffer(44 + n * ch * 2), v = new DataView(ab);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + n * ch * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, ch, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * ch * 2, true);
  v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, n * ch * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < ch; k++) {
      const s = Math.max(-1, Math.min(1, chans[k][i] * gain));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
  }
  return new Blob([ab], { type: "audio/wav" });
}

async function downloadWav() {
  const btn = $("wavBtn");
  btn.disabled = true;
  btn.textContent = "Rendering…";
  try {
    const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const sr = 44100;
    const off = new OAC(2, Math.ceil(sr * (state.piece.duration_sec + 3.5)), sr);
    schedule(off, makeChain(off), state.piece, 0.05);
    const buf = await off.startRendering();
    download(encodeWav(buf), pieceName() + ".wav");
  } catch (e) {
    toast("Could not create the WAV file in this browser.", true);
  } finally {
    btn.disabled = false;
    btn.textContent = "⬇ WAV";
  }
}

// ---------- History ----------
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch (e) { return []; }
}

function saveHistory(list) {
  try { localStorage.setItem(HKEY, JSON.stringify(list)); } catch (e) { /* storage unavailable */ }
}

function addHistory(item) {
  const list = loadHistory().filter((h) => !(
    h.seed === item.seed && h.mood === item.mood && h.key === item.key &&
    h.instrument === item.instrument && h.bars === item.bars && !!h.drums === !!item.drums));
  list.unshift(item);
  saveHistory(list.slice(0, 8));
  renderHistory();
}

function renderHistory() {
  const box = $("hist");
  box.innerHTML = "";
  const list = loadHistory();
  if (!list.length) {
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "Pieces you generate will appear here.";
    box.appendChild(e);
    return;
  }
  list.forEach((h) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "hist-item";
    b.textContent = `${cap(h.mood)} · ${h.key} · ${cap(h.instrument)} · ${h.bars} bars${h.drums ? " · drums" : ""} · #${h.seed}`;
    b.onclick = () => { applyParams(h); generate(h); };
    box.appendChild(b);
  });
}

// ---------- Start-up ----------
async function init() {
  let keys = KEYS_DEFAULT, instruments = INSTR_DEFAULT;
  try {
    const [opts, health] = await Promise.all([api("/api/options"), api("/api/health")]);
    keys = opts.keys; instruments = opts.instruments;
    if (health.model_loaded) setBadge("ok", "AI model online");
    else setBadge("warn", "Backup generator");
  } catch (e) {
    setBadge("bad", "Server offline");
    toast(e.message, true);
  }
  fillSelect("key", keys, "C");
  fillSelect("instrument", instruments, "piano");
  renderHistory();
  drawRoll(-1);
}

document.querySelectorAll(".mood").forEach((b) => {
  b.addEventListener("click", () => setMood(b.dataset.mood));
});

$("tempoAuto").addEventListener("change", () => {
  const auto = $("tempoAuto").checked;
  $("tempo").disabled = auto;
  $("tempoOut").textContent = auto ? "Auto" : $("tempo").value + " BPM";
});
$("tempo").addEventListener("input", () => { $("tempoOut").textContent = $("tempo").value + " BPM"; });
$("creativity").addEventListener("input", () => {
  state.creativityTouched = true;
  $("creativityOut").textContent = creativityLabel(parseFloat($("creativity").value));
});

$("generate").addEventListener("click", () => generate());
$("playBtn").addEventListener("click", async () => {
  try { if (state.playing) pause(); else await play(); }
  catch (e) { toast("Audio could not start in this browser.", true); }
});
$("stopBtn").addEventListener("click", () => stop());
$("midiBtn").addEventListener("click", downloadMidi);
$("wavBtn").addEventListener("click", downloadWav);
$("clearHist").addEventListener("click", () => { saveHistory([]); renderHistory(); });

window.addEventListener("resize", () => drawRoll(state.lastPos));
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body && state.piece) {
    e.preventDefault();
    $("playBtn").click();
  }
});

init();