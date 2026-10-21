import { ARMS, FixedDefault, Thompson, makeReader, makeRng, populationProbs } from "./bandit.js";

const SESSIONS = 60;
const PASSAGES = [
  "The lighthouse keeper climbed the stairs every evening. At the top, she wiped the glass and lit the lamp. Ships far out at sea saw the light and knew the rocks were near. In the morning, she wrote down every ship she had seen.",
  "Bees do a small dance to tell each other where flowers are. The angle of the dance points toward the food. The length of the dance shows how far away it is. Other bees watch closely, then fly straight to the flowers.",
  "A seed needs water, warmth and time. First a tiny root pushes down into the soil. Then a green shoot reaches up toward the light. After a few weeks, the first real leaves open.",
  "The old train station had a clock that never stopped. Travelers checked it before every journey. One winter the clock froze, and nobody knew what time it was. The station master fixed it with a hair dryer and a lot of patience.",
  "Octopuses can change color in less than a second. They use this to hide from danger and to surprise their food. Some can even change the bumps on their skin to look like rocks. Scientists are still learning how they do it.",
];
const PROFILE_LABELS = { difficulty: "Finds decoding hard", confident: "Confident reader", phone: "Reads on a phone" };
const LABEL = {
  font: { standard: "Standard font", dyslexia_friendly: "Dyslexia-friendly font (Lexend)" },
  spacing: { normal: "Normal spacing", wide: "Wide spacing" },
  chunking: { full_page: "Full page", short_chunks: "Short chunks" },
};
const shortName = (a) =>
  `${a.font === "dyslexia_friendly" ? "Lexend" : "Standard"} · ${a.spacing} · ${a.chunking === "short_chunks" ? "chunks" : "full"} · ${a.speed}x`;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- shared UI

function profileChips(container, profile, onChange) {
  container.innerHTML = "";
  for (const [key, label] of Object.entries(PROFILE_LABELS)) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(profile[key]));
    b.onclick = () => {
      profile[key] = !profile[key];
      b.setAttribute("aria-pressed", String(profile[key]));
      onChange();
    };
    container.appendChild(b);
  }
}

function sentences(text) {
  return text.match(/[^.!?]+[.!?]/g).map((s) => s.trim());
}

/** Shows a passage in the given settings; short chunks show two sentences at a time. */
function renderPassage(el, tagsEl, chunkEl, arm, text, chunkIndex = 0, onNext = null) {
  el.className = `passage ${arm.font} ${arm.spacing}`;
  tagsEl.innerHTML = [LABEL.font[arm.font], LABEL.spacing[arm.spacing], LABEL.chunking[arm.chunking], `Read-aloud ${arm.speed}x`]
    .map((t) => `<span class="tag">${t}</span>`).join("");
  const parts = sentences(text);
  if (arm.chunking === "short_chunks") {
    const chunks = [];
    for (let i = 0; i < parts.length; i += 2) chunks.push(parts.slice(i, i + 2).join(" "));
    const i = chunkIndex % chunks.length;
    el.innerHTML = `<p>${chunks[i]}</p>`;
    chunkEl.innerHTML = `Chunk ${i + 1} of ${chunks.length}`;
    if (onNext && chunks.length > 1) {
      const next = document.createElement("button");
      next.className = "btn ghost";
      next.textContent = "Next chunk";
      next.onclick = onNext;
      chunkEl.appendChild(next);
    }
  } else {
    el.innerHTML = `<p>${parts.join(" ")}</p>`;
    chunkEl.textContent = "";
  }
}

function renderBars(el, estimates, truth = null, bestArm = null) {
  const top = estimates.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 8);
  el.innerHTML = top.map(([v, i]) => `
    <div class="bar ${i === bestArm ? "best" : ""}">
      <span class="name">${shortName(ARMS[i])}${i === bestArm ? " ★" : ""}</span>
      <span class="track"><span class="fill" style="width:${(v * 100).toFixed(1)}%"></span>
        ${truth ? `<span class="truth" style="left:calc(${(truth[i] * 100).toFixed(1)}% - 1px)" title="true ${(truth[i] * 100).toFixed(0)}%"></span>` : ""}</span>
      <span>${(v * 100).toFixed(0)}%</span>
    </div>`).join("");
}

// ---------------------------------------------------------------- watch it learn

const watch = { profile: { difficulty: true, confident: false, phone: true }, readerSeed: 7, timer: null };

function resetWatch() {
  const reader = makeReader(watch.profile, makeRng(watch.readerSeed));
  const prior = populationProbs(reader.context);
  watch.reader = reader;
  watch.session = 0;
  watch.lastArm = null;
  watch.runners = [
    { s: new FixedDefault(), color: "--c-fixed" },
    { s: new Thompson(makeRng(101), null, "Thompson (from scratch)"), color: "--c-ts" },
    { s: new Thompson(makeRng(202), prior, "Thompson + similar readers"), color: "--c-pop" },
  ].map((r, k) => ({ ...r, outcomes: makeRng(500 + k), regret: [0], passed: 0 }));
  stopWatch();
  drawWatch();
}

function stepWatch() {
  if (watch.session >= SESSIONS) return stopWatch();
  const { reader } = watch;
  for (const r of watch.runners) {
    const arm = r.s.choose();
    const reward = r.outcomes.uniform() < reader.probs[arm] ? 1 : 0;
    r.s.update(arm, reward);
    r.passed += reward;
    r.regret.push(r.regret.at(-1) + reader.bestProb - reader.probs[arm]);
    if (r === watch.runners.at(-1)) watch.lastArm = arm;
  }
  watch.session += 1;
  drawWatch();
  if (watch.session >= SESSIONS) stopWatch();
}

function startWatch() {
  if (watch.session >= SESSIONS) resetWatch();
  const tick = () => { stepWatch(); if (watch.timer) watch.timer = setTimeout(tick, 1000 / Number($("speed").value)); };
  watch.timer = setTimeout(tick, 0);
  $("play").textContent = "Pause";
}

function stopWatch() {
  clearTimeout(watch.timer);
  watch.timer = null;
  $("play").textContent = watch.session >= SESSIONS ? "Replay" : "Play";
}

function drawChart() {
  const svg = $("chart"), W = 640, H = 240, L = 44, R = 12, T = 12, B = 30;
  // Round the axis up to a clean number (1, 2, 5, 10, 20, ...) so the labels stay readable.
  const raw = Math.max(1, ...watch.runners.map((r) => r.regret.at(-1)));
  const mag = 10 ** Math.floor(Math.log10(raw));
  const maxY = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw);
  const x = (i) => L + (i / SESSIONS) * (W - L - R);
  const y = (v) => T + (1 - v / maxY) * (H - T - B);
  const grid = css("--line"), muted = css("--muted");
  let out = "";
  for (let k = 0; k <= 4; k++) {
    const v = (maxY * k) / 4;
    out += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="${grid}"/>`;
    out += `<text x="${L - 6}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="${muted}">${Number.isInteger(v) ? v : v.toFixed(maxY < 4 ? 2 : 1)}</text>`;
  }
  for (const s of [0, 20, 40, 60]) out += `<text x="${x(s)}" y="${H - 10}" font-size="11" text-anchor="middle" fill="${muted}">${s}</text>`;
  out += `<text x="${(L + W - R) / 2}" y="${H}" font-size="11" text-anchor="middle" fill="${muted}">session</text>`;
  for (const r of watch.runners) {
    const pts = r.regret.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    out += `<polyline points="${pts}" fill="none" stroke="${css(r.color)}" stroke-width="2.5" stroke-linejoin="round"/>`;
  }
  svg.innerHTML = out;
  $("legend").innerHTML = watch.runners.map((r) =>
    `<span style="--c:var(${r.color})">${r.s.name}: regret ${r.regret.at(-1).toFixed(2)}, passed ${r.passed}/${watch.session}</span>`).join("");
}

function drawWatch() {
  const pop = watch.runners.at(-1);
  const arm = ARMS[watch.lastArm ?? pop.s.estimates().indexOf(Math.max(...pop.s.estimates()))];
  renderPassage($("watch-passage"), $("watch-tags"), $("watch-chunk"), arm, PASSAGES[watch.session % PASSAGES.length], watch.session);
  const reveal = $("reveal").checked;
  renderBars($("bars"), pop.s.estimates(), reveal ? watch.reader.probs : null, reveal ? watch.reader.best : null);
  $("session-stat").innerHTML = `Session <b>${watch.session}</b> of ${SESSIONS}` +
    (reveal ? ` · this reader's best: <b>${shortName(ARMS[watch.reader.best])}</b> (${(watch.reader.bestProb * 100).toFixed(0)}%)` : "");
  drawChart();
}

profileChips($("watch-profile"), watch.profile, resetWatch);
$("new-reader").onclick = () => { watch.readerSeed += 1; resetWatch(); };
$("play").onclick = () => (watch.timer ? stopWatch() : startWatch());
$("step").onclick = () => { stopWatch(); stepWatch(); };
$("reset").onclick = resetWatch;
$("reveal").onchange = drawWatch;

// ---------------------------------------------------------------- be the reader

const be = { profile: { difficulty: false, confident: false, phone: false }, rng: makeRng(Date.now() % 1e9) };

function resetBe() {
  be.strategy = new Thompson(be.rng, populationProbs([1, be.profile.difficulty, be.profile.confident, be.profile.phone].map(Number)), "you");
  be.taps = 0; be.easy = 0; be.passage = 0;
  nextBe();
}

function nextBe() {
  be.arm = be.strategy.choose();
  be.chunk = 0;
  drawBe();
}

function drawBe() {
  const arm = ARMS[be.arm];
  renderPassage($("be-passage"), $("be-tags"), $("be-chunk"), arm, PASSAGES[be.passage % PASSAGES.length], be.chunk,
    () => { be.chunk += 1; drawBe(); });
  renderBars($("be-bars"), be.strategy.estimates());
  $("be-stat").innerHTML = be.taps
    ? `<b>${be.taps}</b> taps, <b>${be.easy}</b> comfortable. Leaning toward: <b>${shortName(ARMS[be.strategy.estimates().indexOf(Math.max(...be.strategy.estimates()))])}</b>`
    : "No feedback yet. Its first guess comes from readers like you.";
}

function feedback(reward) {
  window.speechSynthesis?.cancel();
  be.strategy.update(be.arm, reward);
  be.taps += 1; be.easy += reward; be.passage += 1;
  nextBe();
}

profileChips($("be-profile"), be.profile, resetBe);
$("be-easy").onclick = () => feedback(1);
$("be-hard").onclick = () => feedback(0);
$("be-reset").onclick = resetBe;
$("be-speak").onclick = () => {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance($("be-passage").textContent);
  u.rate = ARMS[be.arm].speed;
  window.speechSynthesis.speak(u);
};

// ---------------------------------------------------------------- tabs

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
    $("watch").hidden = tab.dataset.tab !== "watch";
    $("be").hidden = tab.dataset.tab !== "be";
    if (tab.dataset.tab !== "watch") stopWatch();
  };
});

resetWatch();
resetBe();
