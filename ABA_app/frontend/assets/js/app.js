const state = { aspect: null, semantics: null, data: null, view: "aba" };

const el = (id) => document.getElementById(id);

async function loadAspects() {
  const res = await fetch("/api/aspects");
  const { aspects } = await res.json();
  const select = el("aspect-select");
  select.innerHTML = aspects.map((a) => `<option value="${a}">${a}</option>`).join("");
  state.aspect = aspects[0];
}

async function loadGraph() {
  el("status").textContent = "Loading...";
  const params = new URLSearchParams();
  if (state.semantics) params.set("semantics", state.semantics);
  const qs = params.toString();
  const res = await fetch(`/api/graph/${state.aspect}${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    el("status").textContent = "";
    el("extensions").innerHTML = `<div class="error-box">${body.detail || res.statusText}</div>`;
    return;
  }
  const data = await res.json();
  state.data = data;
  state.semantics = data.semantics;

  const semSelect = el("semantics-select");
  if (!semSelect.options.length) {
    semSelect.innerHTML = data.available_semantics
      .map((s) => `<option value="${s}">${s}${data.fast_semantics.includes(s) ? "" : " (slow)"}</option>`)
      .join("");
    semSelect.value = data.semantics;
  }

  renderCurrentView();
  renderLegend();
  renderExtensions(data);
  renderPairs(data);
  el("status").textContent = `${data.aba.assumptions.length} assumptions, ${data.aba.rules.length} rules, ${data.aa.arguments.length} arguments, ${data.aa.defeats.length} defeats`;
}

function renderCurrentView() {
  if (!state.data) return;
  if (state.view === "aba") renderAbaView(state.data);
  else renderAaView(state.data);
}

function renderLegend() {
  const legend = state.view === "aba"
    ? [
        ["Head claim (good_X / bad_X)", "#d9a441"],
        ["Assumption (positive-sentiment body literal)", "#3fb37f"],
        ["Assumption (negative-sentiment body literal)", "#e0575b"],
        ["Contrary atom", "#4f5561"],
        ["Support rule", "#3fb37f (edge)"],
        ["Attack-derivation rule (from Task 3 'Yes' vote)", "#e0575b (edge)"],
      ]
    : [
        ["Constructed argument (premise ⊢ conclusion)", "#4f8cff"],
        ["Defeat (derived attack)", "#e0575b (edge)"],
      ];
  el("legend").innerHTML = legend
    .map(([label, color]) => `<li><span class="swatch" style="background:${color.split(" ")[0]}"></span>${label}</li>`)
    .join("");
}

function renderExtensions(data) {
  el("ext-semantics-label").textContent = data.semantics;
  if (!data.extensions.length) {
    el("extensions").innerHTML = `<p class="hint">No extensions.</p>`;
    return;
  }
  el("extensions").innerHTML = data.extensions
    .map((ext, i) => `<div class="ext-set">#${i + 1} (${ext.length}): ${ext.join(", ") || "(empty)"}</div>`)
    .join("");
}

function renderPairs(data) {
  el("pairs").innerHTML = data.pairs
    .map(
      (p) => `<div class="pair-row">
        <span>A</span><span>${p.a} <span class="hint">(&rarr; ${p.original_a})</span></span>
        <span>B</span><span>${p.b}</span>
        <span>Vote</span><span class="pair-vote ${(p.vote || "").toLowerCase()}">${p.vote ?? "?"}</span>
      </div>`
    )
    .join("");
}

el("aspect-select").addEventListener("change", (e) => {
  state.aspect = e.target.value;
  el("semantics-select").innerHTML = ""; // repopulate for new aspect's response, in case it differs
  loadGraph();
});

el("semantics-select").addEventListener("change", (e) => {
  state.semantics = e.target.value;
  loadGraph();
});

el("view-aba").addEventListener("click", () => {
  state.view = "aba";
  el("view-aba").classList.add("active");
  el("view-aa").classList.remove("active");
  renderCurrentView();
  renderLegend();
});

el("view-aa").addEventListener("click", () => {
  state.view = "aa";
  el("view-aa").classList.add("active");
  el("view-aba").classList.remove("active");
  renderCurrentView();
  renderLegend();
});

(async function init() {
  await loadAspects();
  await loadGraph();
})();
