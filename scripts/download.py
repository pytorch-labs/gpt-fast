# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from requests.exceptions import HTTPError
import subprocess

def download_tinyllamas(repo_id: str, local_dir: str) -> None:
    try:
        model_name = repo_id.split("/")[-1]
        # Download model weight
        weight_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/" + model_name + ".pt"
        weight_dst_path = os.path.join(local_dir, "model.pth")
        subprocess.run(["wget", weight_url, "-O", weight_dst_path], check=True)
        # Download tokenizer model
        tokenizer_url = "https://github.com/karpathy/llama2.c/raw/master/tokenizer.model"
        tokenizer_dst_path: str = os.path.join(local_dir, "tokenizer.model")
        subprocess.run(["wget", tokenizer_url, "-O", tokenizer_dst_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {repo_id}: {e}")

def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    if "stories" in repo_id:
        download_tinyllamas(repo_id, f"checkpoints/{repo_id}")
        return
    try:
        snapshot_download(repo_id, local_dir=f"checkpoints/{repo_id}", local_dir_use_symlinks=False, token=hf_token)
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
