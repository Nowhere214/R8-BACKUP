"""Experimental history drafting inside the mlx-lm streaming wrapper.

mlx-lm 0.27.1 has no public historical-proposal hook. An isolated copy of its
unchanged stream_generate function dispatches to this token generator instead
of the small-model generator. No installed files or module globals are patched.
Only greedy decoding and standard, unquantized KVCache are supported here.
"""

from dataclasses import dataclass
from importlib import import_module
from types import FunctionType

from .adaptive_decode import HistoryIndex


@dataclass
class HistoryAPIStats:
    proposed: int = 0
    accepted: int = 0
    verification_rounds: int = 0
    ordinary_rounds: int = 0
    trimmed_tokens: int = 0
    replay_forwards: int = 0
    target_forwards: int = 0
    lookup_s: float = 0.0
    trim_s: float = 0.0
    final_cache_tokens: int = 0


def historical_generate_step(
    prompt,
    model,
    *,
    history,
    stats,
    draft_length=16,
    ngram=4,
    max_tokens=256,
    prefill_step_size=2048,
):
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import KVCache, make_prompt_cache, trim_prompt_cache
    from time import perf_counter

    if max_tokens < 1 or draft_length < 1 or prefill_step_size < 1:
        raise ValueError("Positive token, draft and prefill limits required")
    if len(prompt) == 0:
        raise ValueError("Nonempty prompt required")
    cache = make_prompt_cache(model)
    if any(type(layer) is not KVCache for layer in cache):
        raise TypeError("Only standard unquantized KVCache is supported")
    prompt_length = len(prompt)
    emitted = []
    index = HistoryIndex(history, n=ngram)
    current = index.add_sequence([])

    def trim(count):
        if count:
            started = perf_counter()
            removed = trim_prompt_cache(cache, count)
            stats.trim_s += perf_counter() - started
            if removed != count:
                raise RuntimeError("Cache trim did not remove the requested suffix")
            stats.trimmed_tokens += count

    with mx.stream(generation_stream):
        # Match the ordinary API prefill partition: leave one pending input.
        pending = prompt.astype(mx.uint32)
        while pending.size > 1:
            count = min(prefill_step_size, pending.size - 1)
            model(pending[:count][None], cache=cache)
            stats.target_forwards += 1
            mx.eval([layer.state for layer in cache])
            pending = pending[count:]
            mx.clear_cache()
        try:
            while len(emitted) < max_tokens:
                lookup_started = perf_counter()
                candidate = index.propose(
                    emitted, min(draft_length, max_tokens - len(emitted) - 1)
                )
                stats.lookup_s += perf_counter() - lookup_started
                proposed = len(candidate)
                inputs = mx.concatenate([pending, mx.array(candidate, mx.uint32)])
                logits = model(inputs[None], cache=cache)[:, -(proposed + 1) :, :]
                stats.target_forwards += 1
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                predictions = mx.argmax(logprobs, axis=-1).reshape(-1)
                mx.eval(predictions, logprobs)
                predicted = predictions.tolist()
                accepted = 0
                while (
                    accepted < proposed and candidate[accepted] == predicted[accepted]
                ):
                    accepted += 1
                stats.proposed += proposed
                stats.accepted += accepted
                if proposed:
                    stats.verification_rounds += 1
                else:
                    stats.ordinary_rounds += 1
                trim(proposed - accepted)
                block = candidate[:accepted] + [predicted[accepted]]
                for position, token in enumerate(block):
                    emitted.append(token)
                    index.append(current, [token])
                    yield token, logprobs[0, position, :], position < accepted
                pending = mx.array([block[-1]], mx.uint32)
        finally:
            # The API can close this generator early on EOS. Discard unused
            # verified entries too, retaining all emitted tokens but the last.
            keep = prompt_length + max(len(emitted) - 1, 0)
            trim(max(cache[0].offset - keep, 0))
            stats.final_cache_tokens = cache[0].offset


def history_stream_generate(
    model,
    tokenizer,
    prompt,
    *,
    history=(),
    draft_length=16,
    ngram=4,
    stats=None,
    max_tokens=256,
    prefill_step_size=2048,
):
    """Retain stock tokenization, EOS handling, detokenization and response API.

    This is an API-integrated extension, not unmodified stock history support.
    A no-history call uses the stock ordinary token generator without changes.
    """
    module = import_module("mlx_lm.generate")
    history = [list(sequence) for sequence in history]
    if not history:
        yield from module.stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
        )
        return
    stats = stats if stats is not None else HistoryAPIStats()

    def step(prompt, model, draft_model, **kwargs):
        del draft_model
        return historical_generate_step(
            prompt,
            model,
            history=history,
            stats=stats,
            draft_length=draft_length,
            ngram=ngram,
            **kwargs,
        )

    original = module.stream_generate
    namespace = dict(original.__globals__, speculative_generate_step=step)
    stream = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    stream.__kwdefaults__ = original.__kwdefaults__
    yield from stream(
        model,
        tokenizer,
        prompt,
        draft_model=object(),
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
    )
