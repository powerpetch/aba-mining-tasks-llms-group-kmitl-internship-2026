#### PATCHARAKORN ####

"""
Task 3 evaluator, structured to match mu_work's Task3_Evaluation.py:
  - one {category}.xlsx per (model, prompt version, aspect) with one sheet per Test column
    (ID, Vote, LLM_Output, Verify) plus a SUMMARY sheet
  - one EVAL_SUMMARY.xlsx per (model, prompt version, aspect) with ALL + MICRO_BY_CATEGORY sheets
  - one top-level TASK3_VERIFY_MASTER_SUMMARY.xlsx aggregating every discovered prediction file
    (ALL_EVAL_SUMMARY + AVG_BY_MODEL_SHOT_TOPIC_CAT, picking the best-scoring Test per config)

Reads predictions from the "_sheet.xlsx" wide tables written by run_task3.py
(outputs/task3/{model}/{label}/{aspect}/{category}/..._sheet.xlsx) and gold Vote from the
Task 3 gold CSVs via src.task3.load_task3_instances (not a separate hardcoded GT path list,
since that loader already handles this project's column-name variants).

"Shot" in the output columns holds the prompt-version label (e.g. "zero_shot", "contrary_v3"),
generalizing mu_work's 0-shot/1-shot column to this project's 8 prompt variants.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
for p in (REPO_ROOT, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))

from internship.ABA_mining.src import load_paths_config
from internship.ABA_mining.src.task3 import load_task3_instances

PRED_BASE = REPO_ROOT / "outputs" / "task3"
EVAL_BASE = REPO_ROOT / "outputs" / "eval" / "task3"
MASTER_SUMMARY_XLSX = EVAL_BASE / "TASK3_VERIFY_MASTER_SUMMARY.xlsx"

# Positive label per category (mirrors mu_work's YES_POSITIVE_SHEETS / NO_POSITIVE_SHEETS).
# Only Contrary(P)Body(N) has gold data in this project; the others are placeholders for
# if/when the remaining Silver sheets become available (see Task3_Implementation_Report.md).
POSITIVE_LABEL_BY_CATEGORY = {
    "Contrary(P)Body(N)": "Yes",
    "Contrary(N)Body(P)": "Yes",
    "Contrary(P)Body(P)": "No",
    "Contrary(N)Body(N)": "No",
}

def norm_yn(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "Yes"
    if s in {"no", "n", "false", "0"}:
        return "No"
    return None


def classify_verify(gt: str | None, pred: str | None, positive_label: str) -> str | None:
    if gt not in {"Yes", "No"} or pred not in {"Yes", "No"}:
        return None
    if gt == positive_label and pred == positive_label:
        return "TP"
    if gt == positive_label and pred != positive_label:
        return "FN"
    if gt != positive_label and pred == positive_label:
        return "FP"
    return "TN"


def safe_div0(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").fillna(0.0).astype(float)
    den = pd.to_numeric(den, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(np.where(den != 0, num / den, 0.0), index=num.index, dtype=float)


def compute_metrics(vote: pd.Series, pred: pd.Series, positive_label: str) -> dict:
    y_true = vote.map(norm_yn)
    y_pred = pred.map(norm_yn)
    ok = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = int(ok.sum())

    if n == 0:
        return {"N_valid": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy": 0.0}

    negative_label = "No" if positive_label == "Yes" else "Yes"
    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    tn = int(((y_true == negative_label) & (y_pred == negative_label)).sum())
    fp = int(((y_true == negative_label) & (y_pred == positive_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred == negative_label)).sum())

    precision = safe_div0(tp, tp + fp)
    recall = safe_div0(tp, tp + fn)
    f1 = safe_div0(2 * precision * recall, precision + recall)
    accuracy = safe_div0(tp + tn, tp + tn + fp + fn)
    return {"N_valid": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}


def _find_prediction_sheets() -> list[Path]:
    return sorted(PRED_BASE.rglob("*_sheet.xlsx"))


def _parse_sheet_path(path: Path) -> tuple[str, str, str, str] | None:
    """Returns (model, label, aspect, category) from
    outputs/task3/{model}/{label}/{aspect}/{category_token}/{run_base}_sheet.xlsx
    """
    try:
        rel = path.relative_to(PRED_BASE)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5:
        return None
    model, label, aspect, category_token = parts[0], parts[1], parts[2], parts[3]
    # Map the filesystem-safe category token back to a display name for the categories we know.
    category = next(
        (c for c in POSITIVE_LABEL_BY_CATEGORY if re.sub(r"[^\w\-]+", "_", c).strip("_") == category_token),
        category_token,
    )
    return model, label, aspect, category


def evaluate_one(model: str, label: str, aspect: str, category: str, pred_xlsx: Path, task3_dir: Path) -> pd.DataFrame:
    positive_label = POSITIVE_LABEL_BY_CATEGORY.get(category)
    if positive_label is None:
        print(f"  [SKIP] unknown category '{category}' for {pred_xlsx.name} — no positive-label mapping")
        return pd.DataFrame()

    # IDs are only unique WITHIN a category (each category sheet restarts its own ID
    # numbering), so the gold lookup must be scoped to this exact (aspect, category) —
    # otherwise votes from a different category sharing the same aspect would collide.
    # max_per_group=None: predictions may have been generated from a random subset: we need
    # the FULL vote lookup so any ID that shows up in the prediction file resolves correctly,
    # not just whichever subset a sampling cap would keep.
    gt_instances = load_task3_instances(task3_dir, aspects=[aspect], categories=[category], max_per_group=None)
    gt_map = {inst.row_id: inst.gold_vote for inst in gt_instances}
    if not gt_map:
        print(f"  [SKIP] no gold data for aspect '{aspect}' / category '{category}'")
        return pd.DataFrame()

    pdf = pd.read_excel(pred_xlsx, dtype=str).fillna("")
    test_cols = [c for c in pdf.columns if re.match(r"^Test \d+$", str(c))]
    if not test_cols:
        print(f"  [SKIP] no Test N columns in {pred_xlsx.name}")
        return pd.DataFrame()

    out_dir = EVAL_BASE / aspect / model / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"{category}.xlsx"

    summary_rows = []
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for tcol in test_cols:
            tmp = pdf[["ID", tcol]].copy()
            tmp.columns = ["ID", "LLM_Output"]
            tmp["LLM_Output"] = tmp["LLM_Output"].map(norm_yn)
            tmp["Vote"] = tmp["ID"].map(gt_map)
            tmp = tmp[tmp["Vote"].notna()].copy()
            tmp["Verify"] = [classify_verify(g, p, positive_label) for g, p in zip(tmp["Vote"], tmp["LLM_Output"])]
            tmp = tmp[["ID", "Vote", "LLM_Output", "Verify"]]
            tmp.to_excel(writer, sheet_name=tcol, index=False)

            met = compute_metrics(tmp["Vote"], tmp["LLM_Output"], positive_label)
            met.update({"Model": model, "Shot": label, "Topic": aspect, "Category": category,
                        "Test": tcol, "Positive_Label": positive_label})
            summary_rows.append(met)

        summary_df = pd.DataFrame(summary_rows)[
            ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
             "N_valid", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "Accuracy"]
        ]
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)

    print(f"  [OK] {aspect} | {model} | {label} | {category} -> {out_xlsx.relative_to(REPO_ROOT)}")
    return summary_df


def write_eval_summary(aspect: str, model: str, label: str, combined_df: pd.DataFrame) -> None:
    if combined_df.empty:
        return
    out_dir = EVAL_BASE / aspect / model / label
    out_path = out_dir / "EVAL_SUMMARY.xlsx"
    cols = ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
            "N_valid", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "Accuracy"]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        combined_df[cols].to_excel(writer, sheet_name="ALL", index=False)

        group_cols = ["Model", "Shot", "Topic", "Category", "Positive_Label"]
        tot = combined_df.groupby(group_cols, as_index=False)[["N_valid", "TP", "TN", "FP", "FN"]].sum()
        tot["Precision"] = safe_div_series(tot["TP"], tot["TP"] + tot["FP"])
        tot["Recall"] = safe_div_series(tot["TP"], tot["TP"] + tot["FN"])
        tot["F1"] = safe_div_series(2 * tot["Precision"] * tot["Recall"], tot["Precision"] + tot["Recall"])
        tot["Accuracy"] = safe_div_series(tot["TP"] + tot["TN"], tot["TP"] + tot["TN"] + tot["FP"] + tot["FN"])
        tot[group_cols + ["Precision", "Recall", "F1", "Accuracy", "N_valid", "TP", "TN", "FP", "FN"]].to_excel(
            writer, sheet_name="MICRO_BY_CATEGORY", index=False
        )
    print(f"  [EVAL_SUMMARY] {out_path.relative_to(REPO_ROOT)}")


def write_master_summary(all_rows: list[pd.DataFrame]) -> None:
    if not all_rows:
        print("\n[WARN] No evaluated rows — nothing to summarize.")
        return

    master_df = pd.concat(all_rows, ignore_index=True)
    cols = ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
            "N_valid", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "Accuracy"]
    all_eval_summary = master_df[cols].copy()

    avg_metrics = (
        master_df.groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
        .agg({"Precision": "mean", "Recall": "mean", "F1": "mean", "Accuracy": "mean"})
    )

    best_pick = (
        master_df.sort_values(
            by=["Model", "Shot", "Topic", "Category", "Positive_Label", "F1", "Accuracy", "Recall", "Precision"],
            ascending=[True, True, True, True, True, False, False, False, False],
        )
        .groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
        .first()
    )

    best_test_wide = (
        best_pick[["Model", "Shot", "Topic", "Category", "Positive_Label", "Test"]]
        .pivot_table(index=["Model", "Topic", "Category", "Positive_Label"], columns="Shot",
                      values="Test", aggfunc="first")
        .reset_index()
    )
    fixed_cols = ["Model", "Topic", "Category", "Positive_Label"]
    shot_cols = sorted(c for c in best_test_wide.columns if c not in fixed_cols)
    best_test_wide = best_test_wide[fixed_cols + shot_cols].rename(columns={c: f"Best Test ({c})" for c in shot_cols})

    avg_by_model_shot_topic_cat = avg_metrics.merge(
        best_test_wide, on=["Model", "Topic", "Category", "Positive_Label"], how="left"
    ).sort_values(by=["Model", "Shot", "Topic", "Category"]).reset_index(drop=True)

    EVAL_BASE.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(MASTER_SUMMARY_XLSX, engine="openpyxl") as writer:
        all_eval_summary.to_excel(writer, sheet_name="ALL_EVAL_SUMMARY", index=False)
        avg_by_model_shot_topic_cat.to_excel(writer, sheet_name="AVG_BY_MODEL_SHOT_TOPIC_CAT", index=False)

    print(f"\n[MASTER SUMMARY SAVED] {MASTER_SUMMARY_XLSX.relative_to(REPO_ROOT)}")


def main() -> None:
    paths_cfg = load_paths_config(REPO_ROOT)

    sheets = _find_prediction_sheets()
    if not sheets:
        sys.exit(f"[ERROR] No prediction files found under {PRED_BASE} "
                  f"(run run_task3.py first)")

    print(f"Found {len(sheets)} prediction sheet(s) under {PRED_BASE.relative_to(REPO_ROOT)}\n")

    # Group by (model, label, aspect) so EVAL_SUMMARY.xlsx covers every category found for that combo.
    grouped: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    for path in sheets:
        parsed = _parse_sheet_path(path)
        if parsed is None:
            print(f"  [SKIP] unrecognized path structure: {path.relative_to(REPO_ROOT)}")
            continue
        model, label, aspect, category = parsed
        grouped.setdefault((model, label, aspect), []).append((category, path))

    all_rows: list[pd.DataFrame] = []
    for (model, label, aspect), items in sorted(grouped.items()):
        combined = []
        for category, path in items:
            df = evaluate_one(model, label, aspect, category, path, paths_cfg.task3_dir)
            if not df.empty:
                combined.append(df)
        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            write_eval_summary(aspect, model, label, combined_df)
            all_rows.append(combined_df)

    write_master_summary(all_rows)


if __name__ == "__main__":
    main()
