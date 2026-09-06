import pytest

from aster.artifacts import save_model
from aster.models import RuntimeEnsemble


def test_save_model_rejects_unserializable_metadata_before_writing_artifact(tmp_path):
    path = tmp_path / "model.joblib"

    with pytest.raises(TypeError):
        save_model(path, RuntimeEnsemble(), {"bad": object()})

    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".metadata.json").exists()
