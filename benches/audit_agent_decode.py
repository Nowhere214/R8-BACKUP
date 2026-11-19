"""Check real pilot prompts against the installed mlx-lm target implementation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tiny_llm.adaptive_decode import ModelEngine, decode  # noqa: E402


def main():
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from tiny_llm.models import dispatch_model

    path = snapshot_download("Qwen/Qwen2-1.5B-Instruct-MLX", local_files_only=True)
    base, tokenizer = load(path)
    silicon = ModelEngine(dispatch_model("Qwen/Qwen2-1.5B-Instruct-MLX", base, week=2))
    native = ModelEngine(base, native=True)
    traces = [
        json.loads(line)
        for line in (ROOT / "benches/agent_pilot_20261015_run2/turns.jsonl")
        .read_text()
        .splitlines()
    ]
    rows = []
    for trace in traces:
        prompt = trace["prompt_token_ids"]
        a = decode(silicon, prompt, 96, {tokenizer.eos_token_id})
        b = decode(native, prompt, 96, {tokenizer.eos_token_id})
        divergence = next(
            (i for i, (x, y) in enumerate(zip(a.tokens, b.tokens)) if x != y), None
        )
        row = {
            "task_id": trace["task_id"],
            "attempt": trace["attempt"],
            "exact": a.tokens == b.tokens,
            "first_divergence": divergence,
            "silicon_tokens": a.tokens,
            "native_tokens": b.tokens,
            "silicon_text": tokenizer.decode(a.tokens),
            "native_text": tokenizer.decode(b.tokens),
        }
        rows.append(row)
        print(
            {k: row[k] for k in ("task_id", "attempt", "exact", "first_divergence")},
            flush=True,
        )
    output = ROOT / "benches/agent_decode_audit_20261021.json"
    output.write_text(
        json.dumps({"model_revision": Path(path).name, "rows": rows}, indent=2) + "\n"
    )
    print("peak_mlx_gib", mx.get_peak_memory() / 1024**3)


if __name__ == "__main__":
    main()
