"""
M4b: upload adapters and dataset to HF Hub. Requires HF_TOKEN with WRITE
scope (the read-only token used elsewhere in this project can't create
repos) -- generate one at https://huggingface.co/settings/tokens and
either export it as HF_TOKEN or pass --token.

Usage:
    python scripts/upload_to_hub.py --username <your-hf-username> [--private]
"""

import argparse
import os

from huggingface_hub import HfApi, upload_folder, upload_file

ADAPTERS = {
    "sft-1.5b-lora-r64": "outputs/sft-sft-1.5b-lora_r64",
    "sft-7b-qlora-r64": "outputs/sft-sft-qlora-7b-epoch1",
    "dpo-7b-final": "outputs/dpo-qlora-7b-v2",
}
DATASET_DIR = "data"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--token", default=None, help="defaults to HF_TOKEN env var")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)

    for name, local_path in ADAPTERS.items():
        repo_id = f"{args.username}/twscholar-lm-{name}"
        print(f"Creating/uploading {repo_id} ...")
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True, token=args.token)
        card_path = f"model_cards/{name}.md"
        if os.path.exists(card_path):
            upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model", token=args.token)
        upload_folder(
            folder_path=local_path, repo_id=repo_id, repo_type="model", token=args.token,
            allow_patterns=["*.safetensors", "*.json", "*.jinja"],
        )
        print(f"  done: https://huggingface.co/{repo_id}")

    dataset_repo = f"{args.username}/twscholar-lm-dataset"
    print(f"Creating/uploading {dataset_repo} ...")
    api.create_repo(dataset_repo, repo_type="dataset", private=args.private, exist_ok=True, token=args.token)
    upload_file(path_or_fileobj="docs/DATASET_CARD.md", path_in_repo="README.md",
                repo_id=dataset_repo, repo_type="dataset", token=args.token)
    upload_folder(
        folder_path=DATASET_DIR, repo_id=dataset_repo, repo_type="dataset", token=args.token,
        allow_patterns=["raw/*.jsonl", "processed/*.jsonl", "eval/holdout_prompts.jsonl"],
    )
    print(f"  done: https://huggingface.co/datasets/{dataset_repo}")


if __name__ == "__main__":
    main()
