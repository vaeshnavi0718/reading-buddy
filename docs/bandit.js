// Browser port of buddy/settings.py, buddy/readers.py and the Thompson
// sampling strategies, so the demo behaves like the Python experiment.

export const FONTS = ["standard", "dyslexia_friendly"];
export const SPACINGS = ["normal", "wide"];
export const CHUNKINGS = ["full_page", "short_chunks"];
export const SPEEDS = [0.8, 1.0, 1.2];

// Same order as itertools.product(FONTS, SPACINGS, CHUNKINGS, SPEEDS)
export const ARMS = [];
for (const font of FONTS)
  for (const spacing of SPACINGS)
    for (const chunking of CHUNKINGS)
      for (const speed of SPEEDS) ARMS.push({ font, spacing, chunking, speed });

export const DEFAULT_ARM = ARMS.findIndex(
  (a) => a.font === "standard" && a.spacing === "normal" && a.chunking === "full_page" && a.speed === 1.0,
);

const features = (a) => [
  1,
  a.font === "dyslexia_friendly" ? 1 : 0,
  a.spacing === "wide" ? 1 : 0,
  a.chunking === "short_chunks" ? 1 : 0,
  a.speed < 1 ? 1 : 0,
  a.speed > 1 ? 1 : 0,
];
const ARM_FEATURES = ARMS.map(features);

// Rows: bias, dyslexia font, wide spacing, short chunks, slow speech, fast speech
// Cols: bias, decoding difficulty, confident reader, on phone   (see buddy/readers.py)
const POPULATION_EFFECTS = [
  [-0.2, -0.5, 0.6, 0.0],
  [0.0, 0.6, 0.0, 0.0],
  [0.0, 0.5, 0.0, -0.2],
  [0.1, 0.0, -0.5, 0.3],
  [0.0, 0.4, -0.4, 0.0],
  [-0.1, -0.6, 0.3, 0.0],
];
const PERSONAL_SD = 0.4;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

// ---- seeded randomness -------------------------------------------------------

export function makeRng(seed) {
  let t = seed >>> 0;
  const uniform = () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(uniform(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
  };
  // Marsaglia-Tsang gamma sampler, used to sample Beta distributions.
  const gamma = (shape) => {
    if (shape < 1) return gamma(shape + 1) * Math.pow(uniform(), 1 / shape);
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x, v;
      do { x = normal(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      const u = uniform();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };
  const beta = (a, b) => { const x = gamma(a); return x / (x + gamma(b)); };
  return { uniform, normal, beta };
}

// ---- readers -----------------------------------------------------------------

export function contextOf({ difficulty, confident, phone }) {
  return [1, difficulty ? 1 : 0, confident ? 1 : 0, phone ? 1 : 0];
}

/** What works on average for readers with this context (no personal quirks). */
export function populationProbs(context) {
  const w = POPULATION_EFFECTS.map((row) => dot(row, context));
  return ARM_FEATURES.map((f) => sigmoid(dot(f, w)));
}

/** A simulated reader: the population pattern for their context plus personal quirks. */
export function makeReader(profile, rng) {
  const context = contextOf(profile);
  const w = POPULATION_EFFECTS.map((row) => dot(row, context) + rng.normal() * PERSONAL_SD);
  const probs = ARM_FEATURES.map((f) => sigmoid(dot(f, w)));
  const best = probs.indexOf(Math.max(...probs));
  return { context, probs, best, bestProb: probs[best] };
}

// ---- strategies ----------------------------------------------------------------

export class FixedDefault {
  name = "Fixed default";
  choose() { return DEFAULT_ARM; }
  update() {}
  estimates() { return ARMS.map((_, i) => (i === DEFAULT_ARM ? 1 : 0)); }
}

/** Beta belief per setting; sample each belief and pick the best sample. */
export class Thompson {
  constructor(rng, prior = null, name = "Thompson sampling") {
    this.rng = rng;
    this.name = name;
    // prior: starting success-rate guess per setting, worth `strength` pretend sessions
    const strength = 4;
    this.alpha = ARMS.map((_, i) => 1 + (prior ? strength * prior[i] : 0));
    this.beta = ARMS.map((_, i) => 1 + (prior ? strength * (1 - prior[i]) : 0));
  }
  choose() {
    let best = 0, bestSample = -1;
    for (let i = 0; i < ARMS.length; i++) {
      const s = this.rng.beta(this.alpha[i], this.beta[i]);
      if (s > bestSample) { bestSample = s; best = i; }
    }
    return best;
  }
  update(arm, reward) {
    if (reward) this.alpha[arm] += 1;
    else this.beta[arm] += 1;
  }
  estimates() { return this.alpha.map((a, i) => a / (a + this.beta[i])); }
}
