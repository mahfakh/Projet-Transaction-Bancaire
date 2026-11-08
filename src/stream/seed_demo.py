"""Script d'initialisation et d'injection de transactions de démonstration."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.aws.orchestration import FraudDetectionPipeline
from src.stream.generator import TransactionStreamGenerator


def seed_demo_data(count: int = 35) -> None:
    print(f"[Seed] Génération de {count} transactions de démonstration...")
    gen = TransactionStreamGenerator()
    pipeline = FraudDetectionPipeline(mode="local")

    scenarios = ["random", "random", "high_amount", "fraud_burst", "random"]
    alert_count = 0

    for i in range(count):
        scenario = scenarios[i % len(scenarios)]
        tx = gen.generate_event(scenario=scenario)
        if tx:
            res = pipeline.process_transaction(tx)
            if res["has_alert"]:
                alert_count += 1

    print(f"[Seed] Terminé ! {count} transactions injectées, {alert_count} alertes générées pour investigation.")


if __name__ == "__main__":
    seed_demo_data(35)
