# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from requests.exceptions import HTTPError
import sys
from pathlib import Path
from typing import Optional

def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id, token=hf_token)

        for file in repo_files:
            if not file.endswith('.safetensors'):
                print(f"Downloading {file}...")
                hf_hub_download(repo_id, filename=file, cache_dir=f"checkpoints/{repo_id}", use_auth_token=hf_token)
                
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="checkpoints/meta-llama/llama-2-7b-chat-hf", help='Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)
