import pytest

from aster.data import jsonl
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


def test_append_observations_write_failure_preserves_existing_evidence(tmp_path, monkeypatch):
    path = tmp_path / "observations.jsonl"
    existing = '{"existing": true}\n'
    path.write_text(existing, encoding="utf-8")

    real_copyfileobj = jsonl.shutil.copyfileobj

    def fail_after_partial_write(source, destination, *args, **kwargs):
        chunk = source.read(4)
        destination.write(chunk)
        destination.flush()
        raise OSError("simulated append failure")

    monkeypatch.setattr(jsonl.shutil, "copyfileobj", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated append failure"):
        append_observations(path, [_Observation({"new": 1})])

    monkeypatch.setattr(jsonl.shutil, "copyfileobj", real_copyfileobj)
    assert path.read_text(encoding="utf-8") == existing
