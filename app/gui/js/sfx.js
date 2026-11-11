// Sound for the arcade look: the effects and the music. All of it is made here from oscillators and
// noise: no audio files. Nothing sounds until arm() is called (look.js, in the arcade look only) and
// the user has clicked or pressed a key once: browsers start audio no earlier.
const KEY = "vtuber-maker-sound", MUSIC = "vtuber-maker-music", SWITCH = "vtuber-maker-music-on", EFFECTS = "vtuber-maker-sound-on";
let level = 0.5, tune = 0.3, on = true, fx = true, ctx = null, out = null, noise = null, moved = 0;
try { on = localStorage.getItem(SWITCH) !== "off"; fx = localStorage.getItem(EFFECTS) !== "off"; } catch {}
const kept = (key, fallback) => { try { const v = parseFloat(localStorage.getItem(key)); return v >= 0 && v <= 1 ? v : fallback; } catch { return fallback; } };
level = kept(KEY, level); tune = kept(MUSIC, tune);

export const volume = () => level, musicVolume = () => tune;
export function setVolume(v) {
  level = v;
  try { localStorage.setItem(KEY, String(v)); } catch {}
  if (out) out.gain.value = v * 0.5;
  wake();
}
export function setMusic(v) {
  tune = v;
  try { localStorage.setItem(MUSIC, String(v)); } catch {}
  wake(); loud();
  if (ctx) v && on ? band() : rest();
}

function wake() {
  if (!(level && fx) && !(tune && on)) return;
  if (!ctx) {
    try { ctx = new AudioContext(); } catch { return; }
    out = ctx.createGain(); out.gain.value = level * 0.5; out.connect(ctx.destination);
    noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    // the music's own way out: its volume, then a filter that closes while a work page is up
    bus = ctx.createGain(); veil = ctx.createBiquadFilter(); veil.type = "lowpass"; veil.frequency.value = calm ? 850 : 9000;
    bus.connect(veil); veil.connect(ctx.destination); loud();
  }
  if (ctx.state === "suspended") ctx.resume().catch(() => {});
  if (tune && on) band();
}
export function arm() {
  // at once, where the window lets a page sound unasked (the app's own window: server._show); a browser tab
  // holds it until the first click or key, which the listeners below are for
  wake();
  addEventListener("pointerdown", wake, true);
  addEventListener("keydown", wake, true);
  // a window out of sight plays no music; a build's step notes still sound
  document.addEventListener("visibilitychange", loud);
}

// one note: from `f` to `to` Hz over `dur` seconds, starting `at` seconds from now
function tone(f, to, dur, type = "square", gain = 0.2, at = 0) {
  const t = ctx.currentTime + at, o = ctx.createOscillator(), g = ctx.createGain();
  o.type = type; o.frequency.setValueAtTime(f, t);
  if (to !== f) o.frequency.exponentialRampToValueAtTime(to, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.006); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g); g.connect(out); o.start(t); o.stop(t + dur + 0.02);
}
// a sweep of filtered noise: the slash across the screen, the hit under "Avatar ready"
function hiss(from, to, dur, gain = 0.2, at = 0) {
  const t = ctx.currentTime + at, src = ctx.createBufferSource(), pass = ctx.createBiquadFilter(), g = ctx.createGain();
  src.buffer = noise; src.loop = true; pass.type = "bandpass"; pass.Q.value = 1.4;
  pass.frequency.setValueAtTime(from, t); pass.frequency.exponentialRampToValueAtTime(to, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + dur * 0.3); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(pass); pass.connect(g); g.connect(out); src.start(t); src.stop(t + dur + 0.02);
}

const SOUNDS = {
  // the cursor moved to another character, card or list entry
  move: () => tone(1050, 1400, 0.045, "square", 0.07),
  // an ordinary button
  click: () => tone(1500, 900, 0.03, "triangle", 0.12),
  select: () => { tone(660, 660, 0.07, "square", 0.11); tone(990, 990, 0.1, "square", 0.11, 0.06); tone(1980, 1980, 0.08, "triangle", 0.05, 0.06); },
  back: () => tone(520, 300, 0.11, "square", 0.1),
  whoosh: () => hiss(500, 3600, 0.26, 0.16),
  // a build starts
  start: () => { tone(220, 880, 0.3, "sawtooth", 0.08); hiss(300, 2400, 0.3, 0.08); },
  // a build step finished: the pitch climbs with how much of the build is done (0..100)
  step: (share = 0) => { const f = 520 * 2 ** (Math.round(share / 100 * 12) / 12); tone(f, f, 0.09, "square", 0.1); tone(f * 1.5, f * 1.5, 0.14, "triangle", 0.09, 0.07); },
  ready: () => {
    hiss(4000, 200, 0.5, 0.3); tone(110, 55, 0.4, "sine", 0.35);
    [523, 659, 784, 1047, 1319].forEach((f, i) => tone(f, f, 0.16 + (i === 4 ? 0.4 : 0), "square", 0.11, 0.08 + i * 0.07));
  },
};

export function play(name, arg) {
  if (!level || !fx || !ctx || ctx.state !== "running") return;
  // a pointer swept across the roster would otherwise rattle
  if (name === "move") { if (ctx.currentTime - moved < 0.04) return; moved = ctx.currentTime; }
  try { SOUNDS[name]?.(arg); } catch {}
}

// ---------- the music. After the reference video's (measured, not heard: about 134 beats a minute with a
// swung, broken beat, snare on two and four, most of its weight under 250 Hz, around D minor): a
// breakbeat, a sub bass riff on the kick's off-beats, short chord stabs, and a riff over the second
// four bars of eight. Scheduled a little ahead of the clock, sixteenth by sixteenth, so a busy page
// can't make it stumble.
let bus = null, veil = null, timer = null, next = 0, beat = 0, calm = false;
const STEP = 60 / 134 / 4, SWING = 0.17, hz = (m) => 440 * 2 ** ((m - 69) / 12);
// eight bars: Dm9, Dm9, Bb, C7, Dm9, Dm9, Gm7, C7
const ROOTS = [38, 38, 34, 36, 38, 38, 43, 36];
const CHORDS = [[62, 65, 69, 76], [62, 65, 69, 76], [58, 62, 65, 69], [60, 64, 67, 70], [62, 65, 69, 76], [62, 65, 69, 76], [58, 62, 65, 67], [60, 64, 67, 70]];
// the bass riff, by sixteenth: [semitones from the root, sixteenths long]
const BASS = { 0: [0, 3], 3: [0, 1], 6: [12, 1], 10: [0, 2], 13: [-5, 1], 14: [0, 2] };
const STABS = [[3, 6, 11], [2, 7, 11]];
// the riff, a bar each: [sixteenth, note, sixteenths long]
const TUNE = [[[0, 74, 2], [3, 77, 1], [4, 79, 2], [7, 81, 3], [12, 84, 2], [14, 81, 2]], [[0, 79, 3], [4, 77, 2], [6, 74, 4]],
  [[0, 82, 2], [3, 79, 1], [4, 77, 2], [7, 79, 3], [12, 74, 4]], [[0, 76, 2], [3, 79, 1], [4, 84, 4], [10, 79, 2], [12, 76, 4]]];

function voice(midi, t, dur, type, gain, cut, snap) {
  const o = ctx.createOscillator(), g = ctx.createGain();
  o.type = type; o.frequency.value = hz(midi);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.008);
  if (!snap) g.gain.setValueAtTime(gain, t + dur * 0.6);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  if (cut) { const f = ctx.createBiquadFilter(); f.type = "lowpass"; f.frequency.value = cut; o.connect(f); f.connect(g); } else o.connect(g);
  g.connect(bus); o.start(t); o.stop(t + dur + 0.02);
}
function drum(t, kind, gain = 1) {
  const g = ctx.createGain(); g.connect(bus);
  if (kind === "kick") {
    const o = ctx.createOscillator(); o.frequency.setValueAtTime(165, t); o.frequency.exponentialRampToValueAtTime(40, t + 0.11);
    g.gain.setValueAtTime(0.6 * gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.24); o.connect(g); o.start(t); o.stop(t + 0.26);
    return;
  }
  const src = ctx.createBufferSource(), f = ctx.createBiquadFilter();
  const [type, at, loud, long] = { snare: ["bandpass", 1700, 0.3, 0.14], hat: ["highpass", 7500, 0.055, 0.03], open: ["highpass", 6500, 0.07, 0.16] }[kind];
  src.buffer = noise; f.type = type; f.frequency.value = at;
  g.gain.setValueAtTime(loud * gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + long);
  src.connect(f); f.connect(g); src.start(t, Math.random() * 0.5); src.stop(t + long + 0.02);
  // the snare's body under its noise
  if (kind === "snare") { const o = ctx.createOscillator(), b = ctx.createGain(); o.type = "triangle"; o.frequency.setValueAtTime(230, t); o.frequency.exponentialRampToValueAtTime(150, t + 0.08);
    b.gain.setValueAtTime(0.22 * gain, t); b.gain.exponentialRampToValueAtTime(0.0001, t + 0.1); o.connect(b); b.connect(bus); o.start(t); o.stop(t + 0.12); }
}
function sixteenth(at, n) {
  const s = n % 16, bar = Math.floor(n / 16) % 8, chord = CHORDS[bar], root = ROOTS[bar];
  // the off sixteenths come late: the swing
  const t = at + (s % 2 ? STEP * SWING : 0);
  if (s === 0 || s === 10 || (s === 6 && bar % 2) || (s === 13 && bar % 4 === 3)) drum(t, "kick", s === 6 ? 0.7 : 1);
  if (s === 4 || s === 12) drum(t, "snare");
  if ((s === 7 || s === 15) && bar % 2) drum(t, "snare", 0.3);
  if (bar === 7 && s >= 13) drum(t, "snare", 0.5 + (s - 13) * 0.2);
  if (s === 14) drum(t, "open"); else if (s % 2 === 0) drum(t, "hat"); else if (s === 3 || s === 11) drum(t, "hat", 0.5);
  if (s in BASS) {
    const [up, long] = BASS[s];
    voice(root + up, t, STEP * long * 0.95, "sine", 0.34); voice(root + up + 12, t, STEP * long * 0.9, "sawtooth", 0.07, 380);
  }
  if (STABS[bar % 2].includes(s)) for (const m of chord) voice(m, t, STEP * 1.6, "sawtooth", 0.028, 1900, true);
  if (bar >= 4) for (const [when, m, long] of TUNE[bar - 4]) if (when === s) { voice(m, t, STEP * long * 0.9, "square", 0.04, 2400); voice(m + 0.08, t, STEP * long * 0.9, "triangle", 0.06); }
}
function ahead() {
  if (!ctx || ctx.state !== "running" || document.hidden) return;
  // back after a pause (a hidden window, a sleeping computer): pick up from now, on a bar line
  if (next < ctx.currentTime - 0.2) { next = ctx.currentTime + 0.06; beat -= beat % 16; }
  while (next < ctx.currentTime + 0.18) { sixteenth(next, beat); next += STEP; beat += 1; }
}
// A song of the user's own, if there is one beside the app (music.mp3, .ogg or .wav in its folder: the
// server's /music), plays in place of the loop above, from its first sound to its last and round again.
let song, looking = null, player = null;   // song: undefined until looked for, then null (none) or { buffer, from, to }
async function findSong() {
  let found = null;
  try {
    const got = await fetch("/music");
    if (got.status === 200) {
      const buffer = await ctx.decodeAudioData(await got.arrayBuffer()), data = buffer.getChannelData(0);
      // an mp3 starts and ends with a little silence, and a song may end with more: not part of the loop
      let a = 0, b = data.length - 1;
      while (a < b && Math.abs(data[a]) < 0.003) a += 16;
      while (b > a && Math.abs(data[b]) < 0.003) b -= 16;
      found = { buffer, from: a / buffer.sampleRate, to: Math.min(buffer.duration, b / buffer.sampleRate + 0.05) };
    }
  } catch {}
  song = found;
}
function band() {
  if (song === undefined) { looking ||= findSong().then(() => { if (tune && on) band(); }); return; }
  if (!song) { if (!timer) timer = setInterval(ahead, 50); return; }
  if (player) return;
  const g = ctx.createGain(); g.gain.value = 0.9; g.connect(bus);
  player = ctx.createBufferSource(); player.buffer = song.buffer; player.loop = true; player.loopStart = song.from; player.loopEnd = song.to;
  player.connect(g); player.start(0, song.from);
}
// (a song keeps running unheard while the music is off, and comes back where it was)
function rest() { clearInterval(timer); timer = null; }
function loud() {
  if (!bus) return;
  bus.gain.setTargetAtTime(document.hidden || !on ? 0 : tune * (calm ? 0.3 : 0.55), ctx.currentTime, 0.25);
}
// The menus have the music whole; a work page has it low and muffled, as from the next room.
export function mood(name) {
  calm = name === "work";
  if (!veil) return;
  veil.frequency.setTargetAtTime(calm ? 850 : 9000, ctx.currentTime, 0.3);
  loud();
}
// The top bar's switches: the effects, and the music, off and on without losing where their sliders stand.
export const soundOn = () => fx && level > 0;
export function switchSound() {
  fx = !soundOn();
  if (fx && !level) setVolume(0.5);
  try { localStorage.setItem(EFFECTS, fx ? "on" : "off"); } catch {}
  wake();
  return fx;
}
export const musicOn = () => on && tune > 0;
export function switchMusic() {
  on = !musicOn();
  if (on && !tune) tune = 0.3;
  try { localStorage.setItem(SWITCH, on ? "on" : "off"); localStorage.setItem(MUSIC, String(tune)); } catch {}
  wake(); loud();
  if (ctx) on ? band() : rest();
  return on;
}
