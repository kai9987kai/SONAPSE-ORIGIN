"""Upload SONAPSE-ORIGIN platform, architectures, and receipts to Hugging Face."""
from __future__ import annotations

import sys
from huggingface_hub import HfApi


def upload():
    api = HfApi()
    repo_id = "Kai9987kai/SONAPSE-ORIGIN"

    ignore_patterns = [
        ".git*",
        ".pytest_cache*",
        "build*",
        "*.pyc",
        "__pycache__*",
        "scratch*",
        "*.db",
        "*.lock",
    ]

    print(f"Uploading SONAPSE-ORIGIN to Hugging Face repo: {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        commit_message="Release SONAPSE-ORIGIN v0.2.0: Neural Organisation Discovery, Causal Emergence, PCI_LZ, and CTC",
    )
    print("Upload to Hugging Face successfully completed!")
    print("Commit info:", commit_info)


if __name__ == "__main__":
    upload()
