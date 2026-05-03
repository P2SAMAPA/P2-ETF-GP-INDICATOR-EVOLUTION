"""push_results.py — Push results to HuggingFace dataset."""

from __future__ import annotations

import io
import json

from huggingface_hub import HfApi

import config


def push_daily_result(result_dict: dict, universe: str = "ALL") -> None:
    universe_slug = universe.lower().replace("_", "-")
    filename = f"gp_indicator_{config.TODAY}_{universe_slug}.json"
    json_bytes = json.dumps(result_dict, indent=2, default=str).encode("utf-8")

    api = HfApi(token=config.HF_TOKEN)

    # Ensure the dataset repo exists
    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    api.upload_file(
        path_or_fileobj=io.BytesIO(json_bytes),
        path_in_repo=filename,
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"GP results {config.TODAY} — {universe_slug}",
    )
    print(f"Results pushed -> {config.HF_OUTPUT_REPO}/{filename}")
