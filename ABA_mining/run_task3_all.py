#### PATCHARAKORN ####

"""
Runs run_task3.py once per prompt version, so a full sweep across all prompt
variants is a single command instead of 8 separate ones.

Usage:
  python run_task3_all.py llama3.2 --aspects check-in check-out price staff
  python run_task3_all.py llama3.2 --versions zero_shot contrary_v3   # subset
  python run_task3_all.py llama4:scout --n-runs 1 --n 20              # quick test, all versions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = REPO_ROOT / "prompts" / "task3"

ALL_PROMPT_VERSIONS = [
    "zero_shot", "one_shot",
    "contrary_v1", "contrary_v2", "contrary_v3", "contrary_v4", "contrary_v5", "contrary_v6",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 across every prompt version in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_name", nargs="?", default=None,
                        help="Optional positional model alias (forwarded to run_task3.py)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--aspects", nargs="+", default=None,
                        help="Restrict to specific aspects, e.g. --aspects check-in check-out price staff")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--versions", nargs="+", default=ALL_PROMPT_VERSIONS,
                         help=f"Which prompt versions to run (default: all 8: {', '.join(ALL_PROMPT_VERSIONS)})")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    unknown_versions = [v for v in args.versions if not (PROMPTS_DIR / f"{v}.txt").exists()]
    if unknown_versions:
        sys.exit(f"[ERROR] No prompt file for: {unknown_versions} "
                  f"(expected under {PROMPTS_DIR.relative_to(REPO_ROOT)}/)")

    model = args.model or args.model_name

    for i, version in enumerate(args.versions, 1):
        cmd = [sys.executable, str(REPO_ROOT / "run_task3.py")]
        if model:
            cmd.append(model)
        cmd += ["--prompt", f"prompts/task3/{version}.txt", "--n-runs", str(args.n_runs)]
        if args.aspects:
            cmd += ["--aspects", *args.aspects]
        if args.categories:
            cmd += ["--categories", *args.categories]
        if args.n is not None:
            cmd += ["--n", str(args.n)]
        if args.max_pairs_per_category is not None:
            cmd += ["--max-pairs-per-category", str(args.max_pairs_per_category)]
        if args.max_tokens is not None:
            cmd += ["--max-tokens", str(args.max_tokens)]
        if args.no_resume:
            cmd.append("--no-resume")

        print(f"\n{'#' * 70}")
        print(f"[{i}/{len(args.versions)}] Prompt version: {version}")
        print(f"{'#' * 70}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    print(f"\n{'=' * 70}")
    print(f"All {len(args.versions)} prompt version(s) completed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
