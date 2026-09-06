import pytest

from aster.data.jsonl import append_observations


class _Observation:
    def __init__(self, payload):
        self.payload = payload

    def to_jsonable(self):
        return self.payload


def test_append_observations_serialization_failure_does_not_partially_append(tmp_path):
    path = tmp_path / "observations.jsonl"
    path.write_text('{"existing": true}\n', encoding="utf-8")

    observations = [
        _Observation({"ok": 1}),
        _Observation({"bad": object()}),
    ]

    with pytest.raises(TypeError):
        append_observations(path, observations)

    assert path.read_text(encoding="utf-8") == '{"existing": true}\n'
