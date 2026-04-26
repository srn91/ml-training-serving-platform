from __future__ import annotations

import argparse
import json

from app.training import train_and_register
from app.validation import validate_offline_online_parity


def main() -> None:
    parser = argparse.ArgumentParser(description="ML training-serving platform CLI")
    parser.add_argument("command", choices=["train", "validate"])
    args = parser.parse_args()

    if args.command == "train":
        artifacts = train_and_register()
        print(json.dumps({"metrics": artifacts.metrics}, indent=2))
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

