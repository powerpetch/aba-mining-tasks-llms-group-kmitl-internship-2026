from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src import load_model_config, load_paths_config, load_topics_config
from src import build_client
from src import run_task1

INDIVIDUAL_PROMPTS = [
    "prompts/task1/generator_v1.txt",
    "prompts/task1/generator_contrastive1.txt",
    "prompts/task1/generator_contrastive2.txt",
    "prompts/task1/generator_contrastive3.txt",
]

COMBINED_PROMPT = "prompts/task1/generator_combined.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 mining pipeline")
    parser.add_argument(
        "--mode",
        choices=["individual", "combined", "both"],
        default="both",
        help="individual = run each prompt separately; combined = run merged prompt; both = run all (default: both)",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of reviews to process (default: 20)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    topics_cfg = load_topics_config(repo_root)
    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)

    if args.mode in ("individual", "both"):
        print("\n" + "=" * 60)
        print("MODE: individual (each prompt separately)")
        print("=" * 60)
        for prompt_path in INDIVIDUAL_PROMPTS:
            prompt_label = Path(prompt_path).stem
            print(f"\n--- Running with prompt: {prompt_label} ---")
            out_path = run_task1(
                repo_root=repo_root,
                client=client,
                topics_cfg=topics_cfg,
                model_cfg=model_cfg,
                paths_cfg=paths_cfg,
                limit_reviews=args.n,
                prompt_path=prompt_path,
                output_subdir="individual",
            )
            print(f"Wrote: {out_path}")

    if args.mode in ("combined", "both"):
        print("\n" + "=" * 60)
        print("MODE: combined (merged prompt with all examples)")
        print("=" * 60)
        prompt_label = Path(COMBINED_PROMPT).stem
        print(f"\n--- Running with prompt: {prompt_label} ---")
        out_path = run_task1(
            repo_root=repo_root,
            client=client,
            topics_cfg=topics_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            limit_reviews=args.n,
            prompt_path=COMBINED_PROMPT,
            output_subdir="combined",
        )
        print(f"Wrote: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

