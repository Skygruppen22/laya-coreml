import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from laya_coreml import Agent, load

RUNTIME_FILES = (
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "model.mlpackage/Manifest.json",
    "model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
)


@pytest.fixture
def bundle(tmp_path):
    package = tmp_path / "model.mlpackage"
    graph = package / "Data/com.apple.CoreML/model.mlmodel"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"small test graph")
    weight = package / "Data/com.apple.CoreML/weights/weight.bin"
    weight.parent.mkdir()
    weight.write_bytes(b"small test weights")
    (package / "Manifest.json").write_text(
        json.dumps(
            {
                "fileFormatVersion": "1.0.0",
                "rootModelIdentifier": "test-model",
                "itemInfoEntries": {
                    "test-model": {
                        "author": "com.apple.CoreML",
                        "description": "CoreML Model Specification",
                        "name": "model.mlmodel",
                        "path": "com.apple.CoreML/model.mlmodel",
                    }
                },
            }
        )
    )
    (tmp_path / "rl_agent_config.json").write_text(
        json.dumps({"max_len": 16, "temperature": [1.0, 1.0, 1.0]})
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    tokens = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 4}
    Tokenizer(WordLevel(tokens, unk_token="[UNK]")).save(str(tokenizer / "tokenizer.json"))
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
            }
        )
    )
    manifest = {
        "format": "laya-coreml",
        "format_version": 1,
        "shape": {
            "batch_size": 1,
            "max_length": 16,
            "min_length": 16,
            "max_options": 2,
            "flexible": False,
            "lengths": None,
        },
        "files": {
            name: {
                "bytes": (tmp_path / name).stat().st_size,
                "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
            }
            for name in RUNTIME_FILES
        },
    }
    (tmp_path / "coreml_config.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def native_load(monkeypatch):
    calls = []

    class NativeModel:
        def __init__(self, path, *, compute_units):
            calls.append((path, compute_units))
            self.inputs = []

        def predict(self, batch):
            self.inputs.append(batch)
            return {
                "logits": np.array([[0.75, -0.5]], dtype=np.float16),
                "action_logits": np.array([[0.25, -0.25]], dtype=np.float16),
            }

    monkeypatch.setitem(
        sys.modules,
        "coremltools",
        SimpleNamespace(
            models=SimpleNamespace(MLModel=NativeModel),
            ComputeUnit=SimpleNamespace(
                CPU_ONLY="CPU_ONLY",
                CPU_AND_GPU="CPU_AND_GPU",
                CPU_AND_NE="CPU_AND_NE",
                ALL="ALL",
            ),
        ),
    )
    return calls


def rewrite_manifest(bundle, update):
    path = bundle / "coreml_config.json"
    manifest = json.loads(path.read_text())
    update(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("name", RUNTIME_FILES)
def test_changed_runtime_file_rejected_before_native_load(bundle, native_load, name):
    (bundle / name).write_bytes(b"changed file")
    with pytest.raises(ValueError, match="does not match its manifest"):
        Agent(bundle)
    assert native_load == []


@pytest.mark.parametrize("name", RUNTIME_FILES)
def test_missing_runtime_file_rejected_before_native_load(bundle, native_load, name):
    (bundle / name).unlink()
    with pytest.raises(FileNotFoundError):
        Agent(bundle)
    assert native_load == []


@pytest.mark.parametrize("inventory", [None, {}, []])
def test_invalid_file_inventory_rejected_before_native_load(bundle, native_load, inventory):
    rewrite_manifest(bundle, lambda manifest: manifest.update(files=inventory))
    with pytest.raises(ValueError, match="non-empty file inventory"):
        Agent(bundle)
    assert native_load == []


def test_missing_file_inventory_rejected_before_native_load(bundle, native_load):
    rewrite_manifest(bundle, lambda manifest: manifest.pop("files"))
    with pytest.raises(ValueError, match="non-empty file inventory"):
        Agent(bundle)
    assert native_load == []


@pytest.mark.parametrize("name", RUNTIME_FILES)
def test_unlisted_runtime_file_rejected_before_native_load(bundle, native_load, name):
    rewrite_manifest(bundle, lambda manifest: manifest["files"].pop(name))
    with pytest.raises(ValueError, match="missing runtime files"):
        Agent(bundle)
    assert native_load == []


def test_unlisted_additional_package_file_rejected_before_native_load(bundle, native_load):
    (bundle / "model.mlpackage/Data/com.apple.CoreML/weights/extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="missing runtime files"):
        Agent(bundle)
    assert native_load == []


@pytest.mark.parametrize("compute_units", ["cpu", "cpu_gpu", "cpu_ne", "all"])
def test_intact_bundle_loads_and_forwards_without_changing_outputs(
    bundle, native_load, compute_units
):
    agent = load(bundle, compute_units=compute_units)
    batch = {"input_ids": np.zeros((1, 16), dtype=np.int32)}
    logits, action = agent.forward(batch)
    expected_units = {
        "cpu": "CPU_ONLY",
        "cpu_gpu": "CPU_AND_GPU",
        "cpu_ne": "CPU_AND_NE",
        "all": "ALL",
    }
    assert native_load == [(str(bundle / "model.mlpackage"), expected_units[compute_units])]
    assert agent.model.inputs == [batch]
    assert agent.tok.pad_token_id == 4
    np.testing.assert_array_equal(logits, [[0.75, -0.5]])
    np.testing.assert_array_equal(action, [[0.25, -0.25]])
    assert logits.dtype == action.dtype == np.float32
