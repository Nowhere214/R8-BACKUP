from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


OUT_DIR = Path("models/gguf").resolve()

MODELS = [
    ("Qwen/Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen2-1.5B-Instruct-GGUF", "qwen2-1_5b-instruct-q4_k_m.gguf"),
    ("Qwen/Qwen2-7B-Instruct-GGUF", "qwen2-7b-instruct-q4_k_m.gguf"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in MODELS:
        print(f"Downloading {repo_id} / {filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=OUT_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
