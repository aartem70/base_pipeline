#!/usr/bin/env python3
"""
Upload a trained model to HuggingFace Hub.

Creates a new repo (if it doesn't exist) and uploads all model files.

Requirements:
    pip install huggingface_hub python-dotenv

Setup:
    Create a .env file in the same directory:
        HF_TOKEN=hf_your_token_here

Usage:
    # Upload a checkpoint:
    python upload.py --model_dir ./distil-checkpoints/final --repo your-username/my-distilled-model

    # Upload a specific step checkpoint:
    python upload.py --model_dir ./distil-checkpoints/step_5000 --repo your-username/my-distilled-model

    # Make the repo private initially (you must make it public before submission):
    python upload.py --model_dir ./distil-checkpoints/final --repo your-username/my-distilled-model --private
"""

import argparse
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Files to skip uploading (not needed for submission, waste space)
SKIP_FILES = {
    "optimizer.pt",
    "train_state.json",
    "train_config.json",
    "train_metrics.csv",
    "train_curves.png",
}


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained model to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local directory containing the model files")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID (e.g. 'your-username/my-distilled-model')")
    parser.add_argument("--private", action="store_true",
                        help="Create as a private repo (must be made public before submission)")
    parser.add_argument("--commit_message", type=str, default="Upload distilled model",
                        help="Commit message for the upload")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        log.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Check required files exist
    files = os.listdir(args.model_dir)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_config = "config.json" in files

    if not has_config:
        log.error("config.json not found in model directory. Is this a valid model?")
        sys.exit(1)

    if not has_safetensors:
        log.error("No .safetensors files found. Model must be saved with safe_serialization=True.")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.error("HF_TOKEN not found. Add it to .env file: HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    api = HfApi(token=hf_token)

    # Verify authentication
    try:
        user_info = api.whoami()
        log.info(f"Authenticated as: {user_info['name']}")
    except Exception:
        log.error("Invalid HF_TOKEN. Check your token in .env file.")
        sys.exit(1)

    # Create repo
    log.info(f"Creating repo: {args.repo} (private={args.private})...")
    try:
        repo_url = create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=hf_token,
        )
        log.info(f"  Repo ready: {repo_url}")
    except Exception as e:
        log.error(f"Failed to create repo: {e}")
        sys.exit(1)

    # Collect files to upload
    upload_files = []
    skip_count = 0
    for fname in sorted(files):
        fpath = os.path.join(args.model_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname in SKIP_FILES:
            skip_count += 1
            continue
        size_mb = os.path.getsize(fpath) / 1e6
        upload_files.append((fname, fpath, size_mb))

    log.info(f"  Files to upload: {len(upload_files)} (skipping {skip_count} training artifacts)")
    for fname, _, size_mb in upload_files:
        log.info(f"    {fname} ({size_mb:.1f} MB)")

    # Upload
    log.info("Uploading...")
    try:
        api.upload_folder(
            folder_path=args.model_dir,
            repo_id=args.repo,
            repo_type="model",
            commit_message=args.commit_message,
            ignore_patterns=list(SKIP_FILES),
        )
    except Exception as e:
        log.error(f"Upload failed: {e}")
        sys.exit(1)

    # Get the commit SHA (needed for submission)
    try:
        repo_info = api.repo_info(repo_id=args.repo, repo_type="model")
        revision = repo_info.sha
    except Exception:
        revision = "(could not fetch)"

    log.info("")
    log.info("=" * 60)
    log.info("  Upload complete!")
    log.info("=" * 60)
    log.info(f"  Repo:     https://huggingface.co/{args.repo}")
    log.info(f"  Revision: {revision}")
    log.info("")
    log.info("  Next steps:")
    log.info(f"    1. Verify:  python evaluate.py --model-repo {args.repo} --revision {revision}")
    log.info(f"    2. Test KL: python evaluate.py --model-repo {args.repo} --eval --prompts 180")
    if args.private:
        log.info(f"    3. Make public before submission: go to https://huggingface.co/{args.repo}/settings")
    log.info("")


if __name__ == "__main__":
    main()
