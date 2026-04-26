from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.dataset import FEATURE_NAMES
from app.service import load_registered_batch, predict_many
from app.training import train_and_register
from app.validation import validate_offline_online_parity


def _load_batch_records(input_path: str | None, limit: int) -> tuple[str, list[dict[str, float]]]:
    if input_path is None:
        return ("registered_dataset_holdout", load_registered_batch(limit=limit))

    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    normalized = [{name: float(record[name]) for name in FEATURE_NAMES} for record in records]
    return (f"file:{input_path}", normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML training-serving platform CLI")
    parser.add_argument("command", choices=["train", "validate", "batch-score"])
    parser.add_argument("--input", help="path to a JSON file containing a list of records or {\"records\": [...]}")
    parser.add_argument("--limit", type=int, default=5, help="default holdout sample size when --input is omitted")
    args = parser.parse_args()

    if args.command == "train":
        artifacts = train_and_register()
        print(json.dumps({"metrics": artifacts.metrics}, indent=2))
        return

    if args.command == "batch-score":
        source, records = _load_batch_records(args.input, args.limit)
        print(
            json.dumps(
                {
                    "batch_scoring": {
                        "source": source,
                        "records_scored": len(records),
                        "predictions": predict_many(records),
                    }
                },
                indent=2,
            )
        )
        return

    summary = validate_offline_online_parity()
    print(
        json.dumps(
            {
                "validation": {
                    "model_version": summary.model_version,
                    "samples_checked": summary.samples_checked,
                    "max_probability_delta": summary.max_probability_delta,
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
