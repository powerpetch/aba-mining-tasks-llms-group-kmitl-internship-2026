from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


from internship.ABA_mining.src import load_paths_config

paths_cfg = load_paths_config(Path(__file__).resolve().parent)
DEFAULT_INPUT_FILE = paths_cfg.gold_csv

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"



ID_CANDIDATES = ["Column1", "ID", "Id", "id", "No", "no", "Index", "index"]
PHRASE_A_CANDIDATES = ["Assumption", "PhraseA", "Phrase A", "A"]
PHRASE_B_CANDIDATES = ["Proposition", "PhraseB", "Phrase B", "B"]


@dataclass(frozen=True)
class RunConfig:
    input_dir: Path
    output_dir: Path
    ollama_host: str
    model: str
    n_runs: int
    batch_size: int
    sleep_sec: float
    one_shot: bool
    resume: bool


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None


def safe_file_token(value: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sheet"


def aspect_from_filename(path: Path) -> str:
    lower = path.name.lower()
    if "check-out" in lower or "checkout" in lower:
        return "check-out"
    if "check-in" in lower or "checkin" in lower:
        return "check-in"
    if "price" in lower:
        return "price"
    if "staff" in lower:
        return "staff"
    return path.stem[:40]


def normalize_yes_no(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    m = re.search(r"\b(yes|no)\b", value, flags=re.IGNORECASE)
    if not m:
        return ""
    return "Yes" if m.group(1).lower() == "yes" else "No"


def build_prompt(phrase_a: str, phrase_b: str, one_shot: bool) -> str:
    base = (
        "Is phrase B contrary to phrase A? Answer exactly 'Yes' or 'No' only. "
        "Do not provide any extra text.\n\n"
        f"[Question] Phrase B is {phrase_b}. Phrase A is {phrase_a}."
    )
    if not one_shot:
        return base

    example_a = "no_evident_not_prompt_response_staff"
    example_b = "no_one_answer_phone"
    return (
        "Is phrase B contrary to phrase A? Answer exactly 'Yes' or 'No' only. "
        "Do not provide any extra text.\n\n"
        f"[Example] Phrase B is {example_b}. Phrase A is {example_a}. Answer is 'Yes'.\n\n"
        f"[Question] Phrase B is {phrase_b}. Phrase A is {phrase_a}."
    )


def ask_ollama(prompt: str, host: str, model: str) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "top_p": 0.05},
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return (data.get("message", {}).get("content") or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"[ERROR] {type(exc).__name__}: {exc}"


def make_key(row_id: str, phrase_a: str, phrase_b: str) -> str:
    return f"{row_id}||{phrase_a}||{phrase_b}"



def run_sheet(
    *,
    workbook_path: Path,
    sheet_name: str,
    output_dir: Path,
    config: RunConfig,
) -> Path:
    sheet_dir = output_dir / aspect_from_filename(workbook_path) / safe_file_token(sheet_name)
    csv_dir = sheet_dir / "csv"
    log_dir = sheet_dir / "log"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_base = f"task3_{safe_file_token(sheet_name)}"
    out_path = csv_dir / f"{run_base}_run1.csv"


    id_col = detect_column(df, ID_CANDIDATES)
    phrase_a_col = detect_column(df, PHRASE_A_CANDIDATES)
    phrase_b_col = detect_column(df, PHRASE_B_CANDIDATES)
    if not id_col or not phrase_a_col or not phrase_b_col:
        raise ValueError(
            f"Could not find required columns in {workbook_path.name}::{sheet_name}. "
            f"Found: {list(df.columns)}"
        )

    work_df = df[[id_col, phrase_a_col, phrase_b_col]].copy()
    work_df.columns = ["ID", "PhraseA", "PhraseB"]
    work_df = work_df.dropna(subset=["ID", "PhraseA", "PhraseB"])
    work_df = work_df[work_df["ID"].astype(str).str.strip() != ""]

    for run_idx in range(1, config.n_runs + 1):
        run_csv = csv_dir / f"{run_base}_run{run_idx}.csv"
        seen_keys: set[str] = set()
        if config.resume and run_csv.exists():
            try:
                old_df = pd.read_csv(run_csv, dtype=str).fillna("")
                if {"ID", "PhraseA", "PhraseB"}.issubset(old_df.columns):
                    seen_keys = {
                        make_key(str(rid), str(a), str(b))
                        for rid, a, b in zip(old_df["ID"], old_df["PhraseA"], old_df["PhraseB"])
                    }
            except Exception:  # noqa: BLE001
                seen_keys = set()

        rows: list[dict[str, Any]] = []
        for _, row in work_df.iterrows():
            key = make_key(str(row["ID"]), str(row["PhraseA"]), str(row["PhraseB"]))
            if key in seen_keys:
                continue
            prompt = build_prompt(str(row["PhraseA"]), str(row["PhraseB"]), config.one_shot)
            raw_output = ask_ollama(prompt, config.ollama_host, config.model)
            pred = normalize_yes_no(raw_output)
            rows.append(
                {
                    "ID": str(row["ID"]),
                    "Prompt": prompt,
                    "PhraseA": str(row["PhraseA"]),
                    "PhraseB": str(row["PhraseB"]),
                    f"Test {run_idx}": pred,
                    "RawOutput": raw_output,
                }
            )
            if config.batch_size and len(rows) >= config.batch_size:
                break

        if rows:
            out_df = pd.DataFrame(rows)
            if run_csv.exists():
                existing = pd.read_csv(run_csv, dtype=str).fillna("")
                combined = pd.concat([existing, out_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["ID", "PhraseA", "PhraseB"], keep="last")
                combined.to_csv(run_csv, index=False)
            else:
                out_df.to_csv(run_csv, index=False)

        if config.sleep_sec > 0 and rows:
            time.sleep(config.sleep_sec)

    return sheet_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Task 3 contradiction pipeline against Ollama.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing the Task 3 Excel files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV outputs")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Ollama base URL")
    parser.add_argument("--model", default=os.getenv("MODEL", "llama3.2"), help="Ollama model to use")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of repeated runs per sheet")
    parser.add_argument("--batch-size", type=int, default=20, help="How many rows to process before pausing")
    parser.add_argument("--sleep-sec", type=float, default=0.25, help="Pause between processed chunks")
    parser.add_argument("--one-shot", action="store_true", help="Use a 1-shot prompt")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing output files and re-run everything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ollama_host=args.ollama_host,
        model=args.model,
        n_runs=args.n_runs,
        batch_size=args.batch_size,
        sleep_sec=args.sleep_sec,
        one_shot=args.one_shot,
        resume=not args.no_resume,
    )

    df = pd.read_csv(DEFAULT_INPUT_FILE)
    print(f"Discovered {len(input_files)} Excel file(s) in {config.input_dir}")

    for workbook_path in input_files:
        print(f"Processing {workbook_path.name}")
        for sheet_name in CATEGORIES:
            if sheet_name in pd.ExcelFile(workbook_path).sheet_names:
                run_sheet(
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    output_dir=config.output_dir,
                    config=config,
                )

    print(f"Outputs written to {config.output_dir}")


if __name__ == "__main__":
    main()
