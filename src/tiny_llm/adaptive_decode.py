"""Greedy decoding with historical proposals and measured-cost selection.

Historical retrieval is an n-gram baseline, not a new drafting algorithm.
All proposals must be verified by the unchanged target model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from time import perf_counter


class ModelEngine:
    def __init__(self, model, native=False):
        self.model = model
        self.native = native

    def cache(self):
        if self.native:
            from mlx_lm.models.cache import make_prompt_cache

            return make_prompt_cache(self.model)
        from .kv_cache import TinyKvFullCache

        return [TinyKvFullCache() for _ in range(self.model.num_hidden_layers)]

    def predict(self, tokens, offset, cache, last_only=False):
        import mlx.core as mx

        x = mx.array(tokens, dtype=mx.int32)[None, :]
        logits = (
            self.model(x, cache=cache) if self.native else self.model(x, offset, cache)
        )
        if last_only:
            logits = logits[:, -1:, :]
        predictions = mx.argmax(logits, axis=-1)
        mx.eval(predictions)
        return predictions.reshape(-1).tolist()

    def rewind(self, cache, count):
        if count <= 0:
            return
        for layer in cache:
            if self.native:
                removed = layer.trim(count)
                if removed != count:
                    raise RuntimeError(
                        "Native cache could not trim verification suffix"
                    )
            else:
                layer.rewind(count)

    def snapshot(self, cache):
        if self.native:
            raise ValueError("Speculative snapshot requires the SiliconLLM cache")
        return [(layer.key_values, layer.offset) for layer in cache]

    def restore(self, cache, snapshot):
        for layer, (values, offset) in zip(cache, snapshot):
            layer.key_values = values
            layer.offset = offset


class HistoryIndex:
    def __init__(self, sequences=(), n=4):
        self.n = n
        self.sequences = []
        self.index = defaultdict(list)
        for sequence in sequences:
            self.add_sequence(sequence)

    def add_sequence(self, tokens):
        seq_id = len(self.sequences)
        self.sequences.append([])
        self.append(seq_id, tokens)
        return seq_id

    def append(self, seq_id, tokens):
        sequence = self.sequences[seq_id]
        for token in tokens:
            sequence.append(token)
            if len(sequence) >= self.n:
                position = len(sequence) - self.n
                self.index[tuple(sequence[position:])].append((seq_id, position))

    def propose(self, prefix, limit):
        if len(prefix) < self.n:
            return []
        hits = self.index.get(tuple(prefix[-self.n :]), [])
        candidates = []
        for seq_id, start in reversed(hits[-32:]):
            begin = start + self.n
            candidate = tuple(self.sequences[seq_id][begin : begin + limit])
            if candidate:
                candidates.append(candidate)
        if not candidates:
            return []
        counts = Counter(candidates)
        return list(
            max(counts, key=lambda candidate: (counts[candidate], len(candidate)))
        )


@dataclass
class CostPolicy:
    baseline_s: float
    verification_s: dict[int, float]
    acceptance: float = 0.8
    margin: float = 0.05
    lookup_s: float = 0.0001

    def choose(self, candidate_length):
        best_k = 0
        best_cost = self.baseline_s * (1 - self.margin)
        for k, verify_s in sorted(self.verification_s.items()):
            if k > candidate_length:
                continue
            expected_tokens = 1 + sum(self.acceptance**j for j in range(1, k + 1))
            recovery_tokens = sum(
                (j + 1) * self.acceptance**j * (1 - self.acceptance) for j in range(k)
            )
            cost = (
                verify_s + self.lookup_s + recovery_tokens * self.baseline_s
            ) / expected_tokens
            if cost < best_cost:
                best_k, best_cost = k, cost
        return best_k

    def update(self, accepted, proposed, lookup_s):
        self.acceptance = 0.7 * self.acceptance + 0.3 * accepted / proposed
        self.lookup_s = 0.8 * self.lookup_s + 0.2 * lookup_s


@dataclass
class DecodeResult:
    tokens: list[int] = field(default_factory=list)
    wall_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    lookup_s: float = 0.0
    decision_s: float = 0.0
    draft_s: float = 0.0
    verification_s: float = 0.0
    cache_s: float = 0.0
    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0
    ordinary_steps: int = 0
    speculative_steps: int = 0
    recovery_steps: int = 0
    fallback_feedback_updates: int = 0
    fallback_feedback_matches: int = 0
    feedback_events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    stop_reason: str = "length"

    def to_dict(self):
        return asdict(self)


def decode(
    engine,
    prompt,
    max_tokens,
    eos_ids,
    *,
    mode="baseline",
    history=(),
    draft_engine=None,
    draft_length=8,
    policy=None,
    ngram=4,
):
    if not prompt or max_tokens < 1:
        raise ValueError("Nonempty prompt and positive max_tokens required")
    modes = {
        "baseline",
        "small",
        "global",
        "structure",
        "adaptive",
        "acceptance",
        "warm_fixed",
        "warm_adaptive",
        "warm_acceptance",
        "warm_feedback",
    }
    if mode not in modes:
        raise ValueError(f"Unknown decode mode: {mode}")
    fallback_feedback = mode == "warm_feedback"
    if mode.startswith("warm_"):
        history = list(history)
        mode = (
            {
                "warm_fixed": "structure",
                "warm_adaptive": "adaptive",
                "warm_acceptance": "acceptance",
                "warm_feedback": "adaptive",
            }[mode]
            if history
            else "baseline"
        )
    if draft_length < 1 or (mode == "small" and draft_engine is None):
        raise ValueError("Invalid drafting configuration")
    if mode == "adaptive" and policy is None:
        raise ValueError("Adaptive decode requires frozen calibration costs")
    result = DecodeResult()
    started = perf_counter()
    cache = engine.cache()
    prefill_started = perf_counter()
    token = engine.predict(prompt, 0, cache, last_only=True)[-1]
    result.prefill_s = perf_counter() - prefill_started
    result.tokens.append(token)
    offset = len(prompt)
    draft_cache = None
    if mode == "small":
        draft_started = perf_counter()
        draft_cache = draft_engine.cache()
        draft_engine.predict(prompt, 0, draft_cache, last_only=True)
        result.prefill_s += perf_counter() - draft_started
    decode_started = perf_counter()
    index = None
    current_sequence = None
    if mode not in {"baseline", "small"}:
        lookup_started = perf_counter()
        sources = [prompt] if mode == "global" else list(history)
        index = HistoryIndex(sources, n=ngram)
        current_sequence = index.add_sequence(result.tokens)
        result.lookup_s += perf_counter() - lookup_started
    recent_acceptance = 0.8
    while len(result.tokens) < max_tokens and token not in eos_ids:
        remaining = max_tokens - len(result.tokens)
        limit = min(draft_length, remaining - 1)
        candidates = []
        lookup_elapsed = 0.0
        if limit > 0 and index is not None:
            lookup_started = perf_counter()
            candidates = index.propose(result.tokens, limit)
            lookup_elapsed = perf_counter() - lookup_started
            result.lookup_s += lookup_elapsed
        decision_started = perf_counter()
        k = min(len(candidates), limit)
        if mode == "small":
            k = max(0, limit)
        elif mode == "adaptive":
            k = policy.choose(k)
        elif mode == "acceptance":
            k = min(8, k) if recent_acceptance >= 0.6 else 0
        result.decision_s += perf_counter() - decision_started
        if k == 0:
            verify_started = perf_counter()
            token = engine.predict([token], offset, cache, last_only=True)[-1]
            result.verification_s += perf_counter() - verify_started
            offset += 1
            result.target_calls += 1
            result.ordinary_steps += 1
            emitted = [token]
            if fallback_feedback and candidates:
                decision_started = perf_counter()
                before = policy.acceptance
                matched = int(token == candidates[0])
                # Reuse the ordinary next-token result; no extra model forward.
                policy.update(matched, 1, lookup_elapsed)
                result.fallback_feedback_updates += 1
                result.fallback_feedback_matches += matched
                result.feedback_events.append(
                    {
                        "output_offset": len(result.tokens),
                        "match": bool(matched),
                        "p_before": before,
                        "p_after": policy.acceptance,
                    }
                )
                result.decision_s += perf_counter() - decision_started
        else:
            candidates = candidates[:k]
            if mode == "small":
                draft_started = perf_counter()
                draft_input = token
                for j in range(k):
                    draft_input = draft_engine.predict(
                        [draft_input], offset + j, draft_cache, last_only=True
                    )[-1]
                    candidates.append(draft_input)
                result.draft_s += perf_counter() - draft_started
            snapshot = engine.snapshot(cache)
            verify_started = perf_counter()
            predictions = engine.predict([token, *candidates], offset, cache)
            verify_elapsed = perf_counter() - verify_started
            result.verification_s += verify_elapsed
            result.target_calls += 1
            accepted = 0
            for candidate, prediction in zip(candidates, predictions):
                if candidate != prediction:
                    break
                accepted += 1
                if candidate in eos_ids:
                    break
            cache_started = perf_counter()
            if accepted < k:
                # Batched FP16 forwards can perturb retained KV values even when
                # the first proposed token is rejected. Restore and replay the
                # retained prefix through the ordinary path on every rejection.
                engine.restore(cache, snapshot)
                emitted = []
                canonical_accepted = 0
                current = token
                for j in range(accepted + 1):
                    current = engine.predict(
                        [current], offset + j, cache, last_only=True
                    )[-1]
                    result.target_calls += 1
                    emitted.append(current)
                    if j < accepted and current == candidates[j]:
                        canonical_accepted += 1
                    else:
                        break
                    if current in eos_ids:
                        break
                accepted = canonical_accepted
                consumed = len(emitted)
                result.recovery_steps += 1
            else:
                emitted = candidates[:accepted]
                if not emitted or emitted[-1] not in eos_ids:
                    emitted.append(predictions[accepted])
                consumed = accepted + 1
            if mode == "small":
                if accepted < k:
                    draft_engine.rewind(draft_cache, k - accepted - 1)
                else:
                    draft_engine.predict(
                        [candidates[-1]], offset + k, draft_cache, last_only=True
                    )
            result.cache_s += perf_counter() - cache_started
            offset += consumed
            token = emitted[-1]
            result.proposed += k
            result.accepted += accepted
            result.speculative_steps += 1
            recent_acceptance = 0.7 * recent_acceptance + 0.3 * accepted / k
            if mode == "adaptive":
                policy.update(accepted, k, lookup_elapsed)
            result.decisions.append(
                {
                    "output_offset": len(result.tokens),
                    "k": k,
                    "accepted": accepted,
                    "verify_s": verify_elapsed,
                }
            )
        result.tokens.extend(emitted)
        if index is not None:
            lookup_started = perf_counter()
            index.append(current_sequence, emitted)
            result.lookup_s += perf_counter() - lookup_started
    result.decode_s = perf_counter() - decode_started
    result.wall_s = perf_counter() - started
    result.stop_reason = "eos" if result.tokens[-1] in eos_ids else "length"
    return result
