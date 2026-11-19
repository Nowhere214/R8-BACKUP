import inspect
from importlib import import_module

import mlx.core as mx
import pytest

from tiny_llm.mlx_lm_history_api import HistoryAPIStats, historical_generate_step


class CountingModel:
    layers = [None]

    def __call__(self, tokens, cache):
        count = tokens.shape[1]
        kv = mx.zeros((1, 1, count, 1))
        cache[0].update_and_fetch(kv, kv)
        next_tokens = (tokens + 1) % 32
        return mx.where(mx.arange(32)[None, None, :] == next_tokens[..., None], 10, -10)


@pytest.mark.parametrize(
    "history", [[], [[4, 5, 6, 7, 8, 9, 10, 11]], [[4, 5, 6, 7, 20, 21]]]
)
def test_trim_history_output_and_cache_match_scalar_counting_model(history):
    stats = HistoryAPIStats()
    tokens = [
        token
        for token, _, _ in historical_generate_step(
            mx.array([1, 2, 3]),
            CountingModel(),
            history=history,
            stats=stats,
            draft_length=4,
            ngram=4,
            max_tokens=9,
        )
    ]
    assert tokens == list(range(4, 13))
    assert stats.final_cache_tokens == 11
    assert stats.replay_forwards == 0
    if history:
        assert stats.verification_rounds > 0
    if history and history[0][4] == 20:
        assert stats.trimmed_tokens > 0


def test_generator_close_trims_unconsumed_verified_tokens():
    stats = HistoryAPIStats()
    generator = historical_generate_step(
        mx.array([1, 2, 3]),
        CountingModel(),
        history=[list(range(4, 15))],
        stats=stats,
        draft_length=8,
        max_tokens=20,
    )
    assert [next(generator)[0] for _ in range(5)] == list(range(4, 9))
    generator.close()
    assert stats.final_cache_tokens == 7
    assert stats.replay_forwards == 0


def test_installed_api_has_expected_dispatch_without_public_history_hook():
    module = import_module("mlx_lm.generate")
    assert "draft_model" in inspect.signature(module.stream_generate).parameters
    assert "speculative_generate_step" in module.stream_generate.__code__.co_names
