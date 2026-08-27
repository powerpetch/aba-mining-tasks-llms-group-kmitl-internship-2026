#### PATCHARAKORN ####

"""
Builds a real py_arg ABAF (flat ABA framework) directly from Task 3's gold data — one
canonical structure that is BOTH what gets rendered to the user AND what py_arg evaluates,
by construction. This is the fix for mu_team's core bug (see
internship/ABA_mining/Doc — the graph shown and the framework evaluated were built
independently and could silently disagree, e.g. synthetic contrary atoms invisible in the
picture). Here there is only one structure; nothing added to it is ever hidden from the
rendered view.

Construction (grounded in the real Task 2/3 ABA data model — see
ABA_mining/Doc/ABA_Labelling_Rules_extracted.txt and Task3_Implementation_Report.md):
  - Each gold row is (Original A, A, B, Vote) for the "Contrary(P)Body(N)" category:
      A        = a body literal from a POSITIVE-sentiment complaint (e.g. easy_check-in)
      Original A = A's auto-derived contrary (no_evident_not_{A})
      B        = a candidate body literal from a NEGATIVE-sentiment complaint
      Vote     = human annotation: does B actually attack A's assumption?
  - Assumptions  = every distinct A and every distinct B (each is itself a body-literal
    assumption in the Task 2 sense).
  - Contrary(A)  = "Original A" column value (already given).
    Contrary(B)  = synthesized as "have_evident_{B}", matching Task 2's own negative-sentiment
    contrary convention (src/task2.py::make_contrary_literals) — B's own default contrary,
    used when B isn't itself under attack by anything else in this slice.
  - Rules:
      good_{aspect} <- A       for every distinct A (Task 2's head-derivation rule)
      bad_{aspect}  <- B       for every distinct B
      Original_A    <- B       for every (A, B) pair where Vote == "Yes" — this is exactly
                                what Task 3 measured: B is real evidence for A's contrary.
  Every assumption gets a contrary (required by ABAF) and the framework is flat (no
  assumption is ever a rule head) — verified by ABAF's own constructor, which raises if not.

NOTE on the current dataset: our own regenerated Task 2 output (llama3.2) uses different
literal normalization than the literals baked into the Task 3 gold CSVs (mu_team's original
extraction) — e.g. "easy_to_check_in" vs "easy_check-in" — so they cannot be cross-referenced
yet. This builder is therefore self-contained within Task 3's gold data (real, not
fabricated), rather than pulling in Task 2 output. Reconciling the two literal vocabularies so
a fresh Task 2 run feeds straight into this builder is a follow-up, not done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from py_arg.aba_classes.aba_framework import ABAF
from py_arg.aba_classes.rule import Rule

from .config import ASPECT_GOLD_FILES, TASK3_GOLD_DIR

PHRASE_A_CONTRARY_CANDIDATES = ["Original Assumption", "Original A"]
PHRASE_A_BODY_CANDIDATES = ["Assumption", "A"]
PHRASE_B_CANDIDATES = ["Proposition", "B"]
VOTE_CANDIDATES = ["Vote"]


@dataclass(frozen=True)
class AbaBuildResult:
    aspect: str
    framework: ABAF
    # Metadata for rendering: every atom's role, kept 1:1 with what's in `framework` — nothing
    # here that isn't also in the framework, and nothing in the framework missing from here.
    atom_roles: dict[str, str]     # atom -> "head" | "assumption_pos" | "assumption_neg" | "contrary"
    pairs: list[dict]              # raw (id, a, b, vote) rows, for the sidebar / audit trail


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None


def build_aba_framework(aspect: str) -> AbaBuildResult:
    if aspect not in ASPECT_GOLD_FILES:
        raise ValueError(f"No gold data for aspect '{aspect}'. Available: {list(ASPECT_GOLD_FILES)}")

    csv_path = TASK3_GOLD_DIR / ASPECT_GOLD_FILES[aspect]
    df = pd.read_csv(csv_path, dtype=str)

    a_contrary_col = _detect_column(df, PHRASE_A_CONTRARY_CANDIDATES)
    a_body_col = _detect_column(df, PHRASE_A_BODY_CANDIDATES)
    b_col = _detect_column(df, PHRASE_B_CANDIDATES)
    vote_col = _detect_column(df, VOTE_CANDIDATES)
    if not all([a_contrary_col, a_body_col, b_col, vote_col]):
        raise ValueError(f"Missing expected columns in {csv_path.name}. Found: {list(df.columns)}")

    df = df[[a_contrary_col, a_body_col, b_col, vote_col]].copy()
    df.columns = ["OriginalA", "A", "B", "Vote"]
    df = df.dropna(subset=["OriginalA", "A", "B"])
    for c in ["OriginalA", "A", "B"]:
        df[c] = df[c].astype(str).str.strip()
    df["Vote"] = df["Vote"].astype(str).str.strip().str.capitalize()
    df = df[(df["A"] != "") & (df["B"] != "")]

    good_head = f"good_{aspect}"
    bad_head = f"bad_{aspect}"

    assumptions: set[str] = set(df["A"]) | set(df["B"])
    contraries: dict[str, str] = {}
    for _, row in df[["A", "OriginalA"]].drop_duplicates().iterrows():
        contraries[row["A"]] = row["OriginalA"]
    for b in set(df["B"]):
        if b not in contraries:
            contraries[b] = f"have_evident_{b}"

    language: set[str] = set(assumptions) | set(contraries.values()) | {good_head, bad_head}

    rules: set[Rule] = set()
    atom_roles: dict[str, str] = {good_head: "head", bad_head: "head"}

    for a in set(df["A"]):
        rules.add(Rule(f"r_good_{a}", {a}, good_head))
        atom_roles[a] = "assumption_pos"
        atom_roles[contraries[a]] = "contrary"

    for b in set(df["B"]):
        rules.add(Rule(f"r_bad_{b}", {b}, bad_head))
        atom_roles.setdefault(b, "assumption_neg")
        atom_roles.setdefault(contraries[b], "contrary")

    pairs: list[dict] = []
    seen_attack_rules: set[tuple[str, str]] = set()
    for i, row in df.reset_index(drop=True).iterrows():
        vote = row["Vote"] if row["Vote"] in {"Yes", "No"} else None
        pairs.append({
            "id": i + 1, "a": row["A"], "original_a": row["OriginalA"], "b": row["B"], "vote": vote,
        })
        if vote == "Yes":
            key = (row["OriginalA"], row["B"])
            if key not in seen_attack_rules:
                seen_attack_rules.add(key)
                rules.add(Rule(f"r_attack_{len(seen_attack_rules)}", {row["B"]}, row["OriginalA"]))

    framework = ABAF(assumptions=assumptions, rules=rules, language=language, contraries=contraries)
    return AbaBuildResult(aspect=aspect, framework=framework, atom_roles=atom_roles, pairs=pairs)
