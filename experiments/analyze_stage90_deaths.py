"""Analyze Stage 90 death bundles into the primary failure taxonomy.

Reads:
  - _docs/stage90_quick_slice.json

Writes:
  - _docs/stage90_quick_slice_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snks.agent.stage90_diagnostics import summarize_failure_buckets

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "_docs"
DEFAULT_INPUT_PATH = DOCS_DIR / "stage90_quick_slice.json"
DEFAULT_OUTPUT_PATH = DOCS_DIR / "stage90_quick_slice_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    bundles = payload.get("bundles", [])
    summary = summarize_failure_buckets(bundles)
    output = {
        "input_path": str(args.input),
        "baseline_reference": payload.get("baseline_reference"),
        "collection_summary": payload.get("summary", {}),
        "analysis_summary": summary,
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(
        f"saved analysis: {args.output} "
        f"(dominant={summary['dominant_bucket']} share={summary['dominant_bucket_share']})"
    )


if __name__ == "__main__":
    main()
