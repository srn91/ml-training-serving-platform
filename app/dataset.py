from __future__ import annotations

import csv
import math
import random
from pathlib import Path


FEATURE_NAMES = [
    "income_k",
    "debt_to_income",
    "credit_score",
    "tenure_months",
    "late_payments_12m",
]


def generate_rows(seed: int = 20260426, rows: int = 2400) -> list[dict[str, float | int]]:
    random.seed(seed)
    dataset: list[dict[str, float | int]] = []
    for _ in range(rows):
        income_k = random.uniform(35.0, 180.0)
        debt_to_income = random.uniform(0.08, 0.65)
        credit_score = random.uniform(540.0, 820.0)
        tenure_months = random.uniform(2.0, 120.0)
        late_payments_12m = random.randint(0, 6)

        score = (
            -2.15
            + (-0.014 * income_k)
            + (4.2 * debt_to_income)
            + (-0.0105 * (credit_score - 680.0))
            + (-0.006 * tenure_months)
            + (0.52 * late_payments_12m)
            + random.gauss(0.0, 0.55)
        )
        probability = 1.0 / (1.0 + math.exp(-score))
        label = 1 if random.random() < probability else 0

        dataset.append(
            {
                "income_k": round(income_k, 4),
                "debt_to_income": round(debt_to_income, 4),
                "credit_score": round(credit_score, 4),
                "tenure_months": round(tenure_months, 4),
                "late_payments_12m": late_payments_12m,
                "defaulted": label,
            }
        )
    return dataset


def write_dataset(rows: list[dict[str, float | int]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FEATURE_NAMES + ["defaulted"],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_dataset(destination: Path) -> list[dict[str, float | int]]:
    with destination.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        parsed_rows: list[dict[str, float | int]] = []
        for row in reader:
            parsed_rows.append(
                {
                    "income_k": float(row["income_k"]),
                    "debt_to_income": float(row["debt_to_income"]),
                    "credit_score": float(row["credit_score"]),
                    "tenure_months": float(row["tenure_months"]),
                    "late_payments_12m": int(row["late_payments_12m"]),
                    "defaulted": int(row["defaulted"]),
                }
            )
    return parsed_rows
