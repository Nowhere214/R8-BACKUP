import pytest

from tiny_llm.adaptive_decode import CostPolicy, HistoryIndex, decode


class FakeEngine:
    def __init__(self, shift=0):
        self.shift = shift

    def cache(self):
        return []

    def predict(self, tokens, offset, cache, last_only=False):
        assert len(cache) == offset
        cache.extend(tokens)
        predictions = [(offset + i + 1 + self.shift) % 7 for i in range(len(tokens))]
        return predictions[-1:] if last_only else predictions

    def rewind(self, cache, count):
        if count:
            del cache[-count:]

    def snapshot(self, cache):
        return list(cache)

    def restore(self, cache, snapshot):
        cache[:] = snapshot


@pytest.mark.parametrize(
    "mode",
    [
        "global",
        "structure",
        "adaptive",
        "acceptance",
        "small",
        "warm_fixed",
        "warm_adaptive",
        "warm_acceptance",
        "warm_feedback",
    ],
)
@pytest.mark.parametrize("eos", [set(), {6}])
def test_all_paths_preserve_greedy_sequence_and_offsets(mode, eos):
    engine = FakeEngine()
    expected = decode(engine, [0, 1], 40, eos)
    result = decode(
        engine,
        [0, 1],
        40,
        eos,
        mode=mode,
        history=[[2, 3, 4, 5, 6, 0, 1] * 5],
        draft_engine=FakeEngine(shift=1),
        draft_length=8,
        policy=CostPolicy(1, {2: 1, 4: 1.2, 8: 2}),
    )
    assert result.tokens == expected.tokens
    assert len(result.tokens) <= 40


def test_matching_small_drafter_aligns_cache_after_full_acceptance():
    expected = decode(FakeEngine(), [0, 1], 40, set())
    result = decode(
        FakeEngine(),
        [0, 1],
        40,
        set(),
        mode="small",
        draft_engine=FakeEngine(),
        draft_length=4,
    )
    assert result.tokens == expected.tokens
    assert result.accepted == result.proposed
    assert result.speculative_steps > 0


def test_partial_acceptance_rewinds_verification_suffix():
    expected = decode(FakeEngine(), [0, 1], 25, set())
    history = [[2, 3, 4, 5, 6, 0, 99, 99] * 4]
    result = decode(
        FakeEngine(),
        [0, 1],
        25,
        set(),
        mode="structure",
        history=history,
        draft_length=4,
    )
    assert result.tokens == expected.tokens
    assert any(0 < event["accepted"] < event["k"] for event in result.decisions)


def test_history_has_no_future_tokens_or_cross_sequence_matches():
    index = HistoryIndex([[1, 2], [3, 4, 5]], n=4)
    assert index.propose([1, 2, 3, 4], 8) == []
    current = index.add_sequence([1, 2, 3, 4])
    assert index.propose([1, 2, 3, 4], 8) == []
    index.append(current, [5, 6])
    assert index.propose([1, 2, 3, 4], 8) == [5, 6]


def test_cost_controller_uses_baseline_when_verification_is_expensive():
    policy = CostPolicy(1, {2: 5, 4: 10})
    assert policy.choose(4) == 0
    cheap = CostPolicy(1, {2: 0.5, 4: 0.7})
    assert cheap.choose(4) > 0
    assert cheap.choose(1) == 0
    cheap.update(0, 4, 0.1)
    assert cheap.acceptance < 0.8


def test_new_ordinary_match_feedback_can_change_a_fallback_decision():
    policy = CostPolicy(1, {2: 1.1, 4: 1.5, 8: 2, 16: 3}, acceptance=0.6, lookup_s=0)
    assert policy.choose(16) == 0
    policy.update(1, 1, 0)
    assert policy.choose(16) > 0


def test_empty_history_and_tail_use_ordinary_decode():
    result = decode(FakeEngine(), [0, 1], 3, set(), mode="structure", history=[])
    assert result.tokens == [2, 3, 4]
    assert result.speculative_steps == 0


@pytest.mark.parametrize(
    "mode", ["warm_fixed", "warm_adaptive", "warm_acceptance", "warm_feedback"]
)
def test_repair_only_modes_use_exact_ordinary_path_without_previous_code(mode):
    expected = decode(FakeEngine(), [0, 1], 40, set())
    result = decode(FakeEngine(), [0, 1], 40, set(), mode=mode, history=[])
    assert result.tokens == expected.tokens
    assert result.speculative_steps == result.recovery_steps == 0
    assert result.lookup_s == 0
    assert result.fallback_feedback_updates == 0


def test_fallback_feedback_resumes_speculation_without_extra_probe_calls():
    kwargs = {
        "history": [[2, 3, 4, 5, 6, 0, 1] * 8],
        "draft_length": 16,
    }

    def policy():
        return CostPolicy(1, {2: 1.1, 4: 1.5, 8: 2, 16: 3}, acceptance=0.6, lookup_s=0)

    expected = decode(FakeEngine(), [0, 1], 40, set())
    stuck = decode(
        FakeEngine(), [0, 1], 40, set(), mode="warm_adaptive", policy=policy(), **kwargs
    )
    resumed = decode(
        FakeEngine(), [0, 1], 40, set(), mode="warm_feedback", policy=policy(), **kwargs
    )
    assert stuck.speculative_steps == 0
    assert resumed.speculative_steps > 0
    assert resumed.fallback_feedback_updates > 0
    assert resumed.fallback_feedback_matches == resumed.fallback_feedback_updates
    assert resumed.feedback_events[0]["p_after"] > 0.6
    assert resumed.tokens == stuck.tokens == expected.tokens
    assert resumed.target_calls == resumed.ordinary_steps + resumed.speculative_steps
    assert resumed.target_calls < stuck.target_calls


def test_fallback_feedback_records_mismatch_and_ignores_missing_candidates():
    expected = decode(FakeEngine(), [0, 1], 12, set())
    policy = CostPolicy(1, {2: 100}, acceptance=0.6)
    result = decode(
        FakeEngine(),
        [0, 1],
        12,
        set(),
        mode="warm_feedback",
        history=[[2, 3, 4, 5, 99, 98]],
        policy=policy,
    )
    assert result.tokens == expected.tokens
    assert result.feedback_events[0]["match"] is False
    assert result.feedback_events[0]["p_after"] == pytest.approx(0.42)
    assert result.fallback_feedback_updates < result.ordinary_steps
    assert result.speculative_steps == 0
    assert result.target_calls == result.ordinary_steps == 11


def test_rejected_batch_restores_retained_state_before_continuing():
    class PerturbedEngine(FakeEngine):
        def predict(self, tokens, offset, cache, last_only=False):
            poisoned = bool(cache and cache[0] == 999)
            predictions = super().predict(tokens, offset, cache, last_only=False)
            if len(tokens) > 1 and offset and predictions[0] != tokens[1]:
                cache[0] = 999
            if poisoned:
                predictions = [(p + 1) % 7 for p in predictions]
            return predictions[-1:] if last_only else predictions

    expected = decode(FakeEngine(), [0, 1], 30, set())
    result = decode(
        PerturbedEngine(),
        [0, 1],
        30,
        set(),
        mode="structure",
        history=[[2, 3, 4, 5, 99, 99, 99, 99]],
        draft_length=4,
    )
    assert result.tokens == expected.tokens
    assert result.recovery_steps > 0


@pytest.mark.parametrize(
    "mode",
    [
        "global",
        "structure",
        "adaptive",
        "acceptance",
        "small",
        "warm_fixed",
        "warm_adaptive",
        "warm_acceptance",
        "warm_feedback",
    ],
)
@pytest.mark.parametrize("seed", range(6))
def test_context_dependent_model_preserves_outputs_across_budgets(mode, seed):
    class ContextEngine(FakeEngine):
        def predict(self, tokens, offset, cache, last_only=False):
            assert len(cache) == offset
            predictions = []
            for token in tokens:
                cache.append(token)
                predictions.append(
                    (
                        sum((i + 3) * value for i, value in enumerate(cache[-5:]))
                        + len(cache)
                        + self.shift
                    )
                    % 13
                )
            return predictions[-1:] if last_only else predictions

    prompt = [seed, 3, seed + 1]
    fixture = decode(ContextEngine(), prompt, 48, set()).tokens
    imperfect = fixture[:12] + [99] + fixture[13:]
    for budget in (1, 2, 7, 31):
        eos = {seed} if seed % 2 else set()
        expected = decode(ContextEngine(), prompt, budget, eos)
        actual = decode(
            ContextEngine(),
            prompt,
            budget,
            eos,
            mode=mode,
            history=[imperfect, fixture],
            draft_length=8,
            draft_engine=ContextEngine(shift=seed + 1),
            policy=CostPolicy(1, {2: 1.1, 4: 1.2, 8: 1.4}),
        )
        assert actual.tokens == expected.tokens
        assert actual.stop_reason == expected.stop_reason
