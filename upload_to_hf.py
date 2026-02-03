#!/usr/bin/env python3
"""
Upload trained Dyck adapter (and optionally merged model) to Hugging Face.

STEP 1 — Create Hugging Face repo and token (do this once):
  1. Go to https://huggingface.co/new and create a new model repo.
     - Name it e.g. "dyck-1.5b-lora" (or your choice).
     - Visibility: Public or Private.
     - License: e.g. Apache-2.0 or MIT.
  2. Create an access token:
     - Go to https://huggingface.co/settings/tokens
     - Click "New token", name it (e.g. "dyck-upload"), scope: "Write" (or "Full").
     - Copy the token and keep it secret.

STEP 2 — Set environment variables (or pass as arguments):
  - HF_TOKEN: your Hugging Face token (required for upload).
  - HF_REPO: your repo id, e.g. "your-username/dyck-1.5b-lora" (or pass via --repo).

STEP 3 — Run this script after training:
  - Upload adapter only (results/):
      set HF_TOKEN=hf_xxx
      set HF_REPO=your-username/dyck-1.5b-lora
      python upload_to_hf.py
  - Upload adapter and merged model (results_merged/ to a second repo):
      python upload_to_hf.py --repo your-username/dyck-1.5b-lora --merged-repo your-username/dyck-1.5b-merged
  - Or: huggingface-cli login   then  python upload_to_hf.py --repo your-username/dyck-1.5b-lora

Usage:
  python upload_to_hf.py [--repo REPO] [--merged-repo REPO] [--adapter-dir DIR] [--no-adapter]

Model card: MODEL_CARD.md is uploaded as README.md to each repo (use --no-model-card to skip).
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# Default paths (same as Train.py)
DEFAULT_ADAPTER_DIR = "results"
DEFAULT_MERGED_DIR = "results_merged"
MODEL_CARD_FILENAME = "MODEL_CARD.md"


def _get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token as _hf_get_token
            return _hf_get_token()
        except Exception:
            pass
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Upload Dyck adapter (and optionally merged model) to Hugging Face."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("HF_REPO"),
        help="Hugging Face repo id (e.g. username/dyck-1.5b-lora). Default: HF_REPO env.",
    )
    parser.add_argument(
        "--merged-repo",
        type=str,
        default=os.environ.get("HF_MERGED_REPO"),
        help="Optional: repo id for merged model. If set, uploads results_merged/ there.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help=f"Directory containing LoRA adapter and tokenizer. Default: {DEFAULT_ADAPTER_DIR}",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=DEFAULT_MERGED_DIR,
        help=f"Directory containing merged model (if --merged-repo). Default: {DEFAULT_MERGED_DIR}",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Skip uploading adapter (use only with --merged-repo).",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repo on Hugging Face if it does not exist (requires token with write).",
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Do not upload MODEL_CARD.md as README.md to the repo(s).",
    )
    args = parser.parse_args()

    token = _get_hf_token()
    if not token and (args.repo or args.merged_repo):
        print(
            "No Hugging Face token found. Set HF_TOKEN or run: huggingface-cli login",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)

    if not args.repo and not args.merged_repo:
        print("Set --repo (or HF_REPO) and/or --merged-repo (or HF_MERGED_REPO).", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

    root = os.path.dirname(os.path.abspath(__file__))
    model_card_path = os.path.join(root, MODEL_CARD_FILENAME)
    upload_model_card = not args.no_model_card and os.path.isfile(model_card_path)

    def _upload_readme(repo_id: str) -> None:
        if not upload_model_card:
            return
        try:
            upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"Model card (README.md) uploaded to {repo_id}")
        except Exception as e:
            print(f"Model card upload warning: {e}")

    # Upload adapter (results/)
    if args.repo and not args.no_adapter:
        adapter_path = os.path.join(root, args.adapter_dir)
        if not os.path.isdir(adapter_path):
            print(f"Adapter dir not found: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        if args.create_repo:
            try:
                create_repo(args.repo, repo_type="model", token=token, exist_ok=True)
            except Exception as e:
                print(f"Create repo warning: {e}")
        print(f"Uploading adapter from {adapter_path} to {args.repo} ...")
        try:
            upload_folder(
                folder_path=adapter_path,
                repo_id=args.repo,
                repo_type="model",
                token=token,
            )
            print(f"Adapter uploaded: https://huggingface.co/{args.repo}")
            _upload_readme(args.repo)
        except Exception as e:
            print(f"Upload failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Upload merged model
    if args.merged_repo:
        merged_path = os.path.join(root, args.merged_dir)
        if not os.path.isdir(merged_path):
            print(f"Merged dir not found: {merged_path}", file=sys.stderr)
            if not args.repo:
                sys.exit(1)
        else:
            if args.create_repo:
                try:
                    create_repo(args.merged_repo, repo_type="model", token=token, exist_ok=True)
                except Exception as e:
                    print(f"Create repo warning: {e}")
            print(f"Uploading merged model from {merged_path} to {args.merged_repo} ...")
            try:
                upload_folder(
                    folder_path=merged_path,
                    repo_id=args.merged_repo,
                    repo_type="model",
                    token=token,
                )
                print(f"Merged model uploaded: https://huggingface.co/{args.merged_repo}")
                _upload_readme(args.merged_repo)
            except Exception as e:
                print(f"Merged upload failed: {e}", file=sys.stderr)
                sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
