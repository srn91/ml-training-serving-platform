from __future__ import annotations

import json
import subprocess
import sys

from app.training import train_and_register


def test_batch_score_cli_uses_registered_holdout_by_default() -> None:
    train_and_register()

    completed = subprocess.run(
        [sys.executable, "-m", "app.cli", "batch-score", "--limit", "3"],
        capture_output=True,
        check=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    batch_scoring = payload["batch_scoring"]
    assert batch_scoring["source"] == "registered_dataset_holdout"
    assert batch_scoring["records_scored"] == 3
    assert len(batch_scoring["predictions"]) == 3
