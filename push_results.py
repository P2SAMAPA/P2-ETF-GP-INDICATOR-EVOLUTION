"""
Push results to Hugging Face dataset.
"""

import json
from huggingface_hub import HfFileSystem
import config

def push_daily_result(result_dict):
    filename = f"gp_indicator_{config.TODAY}.json"
    fs = HfFileSystem(token=config.HF_TOKEN)
    json_str = json.dumps(result_dict, indent=2)
    with fs.open(f"datasets/{config.HF_OUTPUT_REPO}/{filename}", "w") as f:
        f.write(json_str)
    print(f"Results pushed to {config.HF_OUTPUT_REPO}/{filename}")
