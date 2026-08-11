"""End-to-end CPU smoke run.

Skips rather than fails when the data or the base checkpoint are unavailable, so
`pytest` still exits 0 on a fresh clone with no network. The real end-to-end
verification is `python -m codetune run --config configs/smoke.yaml`.
"""

import json
from pathlib import Path

import pytest

from codetune.data import DATA_ROOT
from codetune.train import RunConfig, train_one_run

pytestmark = pytest.mark.slow


def _needs(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not staged; run `python -m codetune prepare`")


def test_smoke_run_writes_a_complete_result(tmp_path):
    _needs(DATA_ROOT / "defect" / "train.jsonl")
    transformers = pytest.importorskip("transformers")
    try:
        transformers.AutoTokenizer.from_pretrained("microsoft/codebert-base")
    except Exception as exc:  # offline, or the hub is unreachable
        pytest.skip(f"codebert-base unavailable: {exc}")

    cfg = RunConfig(
        task="defect",
        method="bitfit",
        epochs=1,
        batch_size=4,
        grad_accum=2,
        max_length=128,
        limit_train=32,
        limit_eval=16,
        device="cpu",
        fp16=False,
        output_dir=str(tmp_path),
    )
    result = train_one_run(cfg, verbose=False)

    out = Path(tmp_path) / "defect" / "bitfit__seed42.json"
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["run_id"] == result["run_id"] == "defect__bitfit__seed42"

    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["data"]["train"]["n_used"] == 32
    assert result["data"]["train"]["subsampled"] is True

    cost = result["cost"]
    assert 0 < cost["trainable_params"] < cost["total_params"]
    assert cost["delta_checkpoint_bytes"] > 0
    assert cost["seconds"] > 0
    # CPU runs report no memory figure rather than a misleading zero.
    assert cost["peak_memory_bytes"] is None
