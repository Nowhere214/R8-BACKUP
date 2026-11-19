import pytest

from tiny_llm.native_history import NativeHistoryEngine

mx = pytest.importorskip("mlx.core")
native_cache = pytest.importorskip("mlx_lm.models.cache")
KVCache = native_cache.KVCache
RotatingKVCache = native_cache.RotatingKVCache


def test_native_snapshot_survives_indexed_assignment_and_restores_offset():
    engine = NativeHistoryEngine(None)
    with mx.stream(mx.cpu):
        layer = KVCache()
        keys = mx.ones((1, 1, 2, 4))
        values = 2 * keys
        layer.update_and_fetch(keys, values)
        snapshot = engine.snapshot([layer])
        layer.keys[..., 0, :] = 99
        layer.update_and_fetch(3 * keys, 4 * keys)
        mx.eval(layer.keys, snapshot[0][0])
        assert float(snapshot[0][0][0, 0, 0, 0].item()) == 1
        engine.restore([layer], snapshot)
        assert layer.offset == 2
        assert float(layer.keys[0, 0, 0, 0].item()) == 1
        layer.keys[..., 0, :] = 77
        assert float(snapshot[0][0][0, 0, 0, 0].item()) == 1


def test_native_adapter_rejects_unsupported_rotating_cache():
    with pytest.raises(TypeError, match="non-rotating"):
        NativeHistoryEngine(None).snapshot([RotatingKVCache(max_size=16)])
