"""push_results.py — Push results to HuggingFace dataset."""

from __future__ import annotations

import json

from huggingface_hub import HfFileSystem

import config


def push_daily_result(result_dict: dict) -> None:
    filename = f"gp_indicator_{config.TODAY}.json"
    fs = HfFileSystem(token=config.HF_TOKEN)
    json_str = json.dumps(result_dict, indent=2, default=str)

    dest = f"datasets/{config.HF_OUTPUT_REPO}/{filename}"
    with fs.open(dest, "w") as f:
        f.write(json_str)
    print(f"Results pushed → {dest}")
