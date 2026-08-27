# ABA App

A from-scratch rebuild of mu_team's `Task4/Project_ABA` web app, addressing the specific bug
you flagged ("the graph is represented wrong with AA and ABA") and covering your listed
next steps: develop the web app, fix the GUI's graph rendering, improve the ABA reasoner, and
document `py_arg`'s real API.

## What was wrong with mu_team's version (verified by reading their code directly)

mu_team's app built **two separate graphs from the same underlying data**: a hand-rolled,
heuristic "argument tree" (drawn to the screen) and a separately re-derived JSON payload (sent
to `py_arg` for evaluation). These could silently disagree — e.g. every assumption without a
real attacker got a synthetic self-contrary injected into the `py_arg` payload so `ABAF()`
wouldn't crash, but that synthetic contrary was **never drawn**, so the picture could show an
assumption as undefended while the framework actually being evaluated had a hidden contrary
for it. There's also no "argument" concept anywhere in their rendering — attack edges are
drawn directly between raw sentence-level atoms, never between constructed arguments — so the
picture is neither a faithful AA graph nor a complete ABA diagram, just an ad hoc hybrid of
both.

**The fix here:** one canonical `ABAF` object (`app/aba_builder.py`) is the single source of
truth for everything — the same object is rendered as the ABA view, converted via `py_arg`'s
own `generate_af()` for the AA view, and passed to `py_arg`'s semantics functions for
extensions. Nothing is computed twice from independently-derived data, so the picture and the
evaluated framework cannot diverge.

## Architecture

- **Backend:** Python + FastAPI, calling `py_arg` (`python-argumentation` on PyPI, import name
  `py_arg`) **in-process** — no subprocess/JSON bridge like mu_team needed (they were bridging
  Python from Node.js; since this backend is already Python, that whole bridge disappears).
- **Data:** reads Task 3's gold CSVs directly from `internship/ABA_mining/Dataset/Task3/` — no
  database, no manual export/import step. Always reflects whatever's on disk in the pipeline
  project.
- **Frontend:** plain HTML/CSS/JS + [Cytoscape.js](https://js.cytoscape.org/) (via CDN, no
  build step), replacing mu_team's hand-rolled SVG renderer. Two explicit views — **ABA**
  (assumptions/rules/contraries, 1:1 with `framework.rules`/`framework.contraries`) and **AA**
  (constructed arguments/defeats, 1:1 with `framework.generate_af()`) — with a toggle, instead
  of one conflated tree.

## How the ABA framework is built (`app/aba_builder.py`)

Grounded directly in the real Task 2/3 data model (see
`ABA_mining/Doc/ABA_Labelling_Rules_extracted.txt`):

- Each gold row is `(Original A, A, B, Vote)`: `A` is a positive-sentiment body literal,
  `Original A` is its auto-derived contrary, `B` is a candidate negative-sentiment body
  literal, `Vote` is the human annotation of whether B attacks A.
- **Assumptions** = every distinct `A` and `B`. **Contraries**: `A`'s is the given `Original A`;
  `B`'s is synthesized as `have_evident_{B}` (matching Task 2's own convention for
  negative-sentiment literals, `src/task2.py::make_contrary_literals`).
- **Rules**: `good_{aspect} <- A` for every `A`, `bad_{aspect} <- B` for every `B`, and —
  the key one — `Original_A <- B` for every pair where `Vote == "Yes"`. That last rule is
  exactly what Task 3 measured: B is real evidence for A's contrary.

**Known limitation:** our freshly-regenerated Task 2 output (llama3.2) uses different literal
normalization than the literals baked into the Task 3 gold CSVs (mu_team's original
extraction) — e.g. `easy_to_check_in` vs `easy_check-in` — so they can't be cross-referenced
yet. This builder is self-contained within Task 3's gold data (real, not fabricated) rather
than pulling in Task 2. Reconciling the two literal vocabularies so a fresh Task 2 run feeds
straight into this builder is a follow-up.

## `py_arg` API notes (for your "study PyArg's API" task)

Verified directly against the installed package, not just by reading mu_team's usage:

- `ABAF(assumptions, rules, language, contraries)` — **enforces flatness itself** (raises
  `ValueError` if any assumption appears as a rule head) and requires every assumption to have
  a contrary. Useful: you don't need to hand-verify these constraints, the constructor does it.
- `abaf.generate_af()` — the real ABA→AA (Dung) instantiation. For every assumption's contrary,
  it recursively constructs *every* possible argument (minimal assumption set) that derives it,
  then computes defeats between all constructed arguments. This is genuinely reusable — every
  semantics function in `py_arg` calls this internally, so it's already the library's own
  single source of truth; we just call it once ourselves and reuse the result.
- **Confirmed upstream bug:** `py_arg.aba_classes.semantics.get_grounded_extensions` — the
  *module* is named for grounded semantics, but the *function inside it* is literally named
  `get_preferred_extensions` (a typo in the installed package itself). This is exactly why
  mu_team's `pyarg_runner.py` had to guess across several possible function names — it's not
  their bug, it's upstream. Wrapped and aliased correctly in `app/pyarg_service.py`.
- **Performance — measured, not assumed:** on a real 48-assumption / 142-argument framework
  (check-in), `grounded` computed in 0.01s (proper least-fixed-point algorithm, always
  terminates, always exactly one extension). `stable` did not finish in 60+ seconds and was
  killed — its algorithm is an unpruned binary-branching search over every AA-level argument
  (`2^142` in the worst case). `admissible`/`conflict_free` enumerate the full assumption
  powerset (`2^48`). **`grounded` is the only semantics reliably fast at this scale**; the
  others are offered but wrapped in a 15-second timeout (`EXTENSION_TIMEOUT_SECONDS` in
  `pyarg_service.py`) that returns a clear error instead of hanging the server.

## Running it

```bash
cd internship/ABA_app
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8123
```
Open `http://127.0.0.1:8123/`. Pick an aspect (check-in / check-out / price / staff — the 4
topics with real human-voted gold data) and a semantics (`grounded` is the safe default; others
are marked "(slow)" in the dropdown). Toggle between ABA and AA views with the buttons at top.

## Verified so far

- `ABAF` construction succeeds and passes its own flatness/contrary validation for check-in
  (48 assumptions, 142 rules).
- `generate_af()` produces 142 arguments / 94 defeats.
- `grounded` extension computes correctly (one extension, 23 assumptions) in 0.01s.
- Full API round-trip tested (`GET /api/graph/check-in` → 200, correct JSON shape).
- Static frontend files served correctly, JS passes syntax checking (`node --check`).
- **Not yet done:** actually opening it in a browser to visually confirm the Cytoscape
  rendering looks right — I don't have browser automation available in this environment, so
  please open `http://127.0.0.1:8123/` yourself and tell me if anything looks off (node
  overlap, missing legend colors, etc.) before we call the GUI fix complete.

## Next steps (matching your task list)

1. **You test it in a browser** and report back anything visually wrong — this closes the loop
   on "fix GUI's implementation."
2. Wire in `check-out`/`price`/`staff` (same builder, already parameterized by aspect — should
   just work, but not yet tested against them).
3. Decide whether/how to reconcile Task 2's regenerated literals with Task 3's gold literals, to
   replace the "self-contained from Task 3 gold only" construction with a fuller pipeline that
   also incorporates fresh Task 2 output.
4. Facility topic (once its remaining 2 categories are fully voted) and the `contrary_v5`-style
   "best prompt" predictions (as an alternative to gold votes) as a second toggle, to compare
   what the reasoner concludes under gold vs. LLM-predicted attacks.
