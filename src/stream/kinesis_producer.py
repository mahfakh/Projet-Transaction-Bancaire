"""Producteur Kinesis : envoie des transactions vers le flux AWS (F1, Architecture B).

Exemples (identifiants AWS Academy dans ~/.aws/credentials) :
    python src/stream/kinesis_producer.py --count 50
    python src/stream/kinesis_producer.py --count 30 --fraud-ratio 0.3 --invalid 2 --attack 1
    python src/stream/kinesis_producer.py --dry-run --count 5     # affiche sans envoyer

Clé de partition = clé du porteur (`user_or_card_key`) : toutes les transactions d'une même
carte passent par le même shard et sont traitées dans l'ordre par la Lambda de scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import get_config


def to_json_safe(value: Any) -> Any:
    """NaN -> None et types numpy -> types Python : le JSON envoyé doit être standard."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_events(count: int, fraud_ratio: float, invalid: int, attacks: int) -> Iterator[Dict[str, Any]]:
    """Construit la séquence d'événements à envoyer (mélange normal / fraude / invalides / attaques)."""
    from src.stream.generator import TransactionStreamGenerator

    gen = TransactionStreamGenerator()
    plan: List[str] = ["normal"] * count + ["invalid"] * invalid + ["attack"] * attacks
    random.shuffle(plan)

    for kind in plan:
        if kind == "invalid":
            # Envoyé tel quel (sans validation locale) pour démontrer le rejet côté AWS -> DLQ
            yield random.choice([
                {"TransactionID": f"INVALID_{int(time.time() * 1000)}", "TransactionAmt": -50.0, "ProductCD": "W"},
                {"TransactionID": f"INVALID_{int(time.time() * 1000)}", "ProductCD": "W"},  # montant absent
                {"TransactionID": f"INVALID_{int(time.time() * 1000)}", "TransactionAmt": "abc"},
            ])
        elif kind == "attack":
            # Scénario « carte compromise » : 5 achats rapides, montants croissants, nouveau terminal
            base = gen.generate_event(scenario="random")
            card_key = f"ATTACK_{random.randint(1000, 9999)}"
            for i in range(5):
                evt = dict(base)
                evt.update(
                    TransactionID=f"TX_{int(time.time() * 1000)}_{random.randint(100, 999)}",
                    user_or_card_key=card_key,
                    TransactionAmt=round(40.0 * (2.2 ** i), 2),
                    DeviceInfo="Windows" if i == 0 else "rv:57.0",
                    event_timestamp=_now_iso(),
                )
                yield evt
        else:
            scenario = "fraud_burst" if random.random() < fraud_ratio else "random"
            evt = gen.generate_event(scenario=scenario)
            if evt:
                evt["event_timestamp"] = _now_iso()
                yield evt


def main() -> None:
    parser = argparse.ArgumentParser(description="Envoie des transactions de démonstration vers Kinesis.")
    parser.add_argument("--count", type=int, default=20, help="Nombre de transactions normales/frauduleuses")
    parser.add_argument("--fraud-ratio", type=float, default=0.2, help="Part de transactions frauduleuses (0-1)")
    parser.add_argument("--invalid", type=int, default=0, help="Nombre d'événements invalides (test de la DLQ)")
    parser.add_argument("--attack", type=int, default=0, help="Nombre de scénarios « carte compromise » (5 tx chacun)")
    parser.add_argument("--delay", type=float, default=0.2, help="Pause entre deux envois (secondes)")
    parser.add_argument("--stream", default=None, help="Nom du flux (défaut : configuration KINESIS_STREAM)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les événements sans les envoyer")
    args = parser.parse_args()

    cfg = get_config()
    stream = args.stream or cfg.kinesis_stream
    kinesis = None
    if not args.dry_run:
        import boto3
        kinesis = boto3.client("kinesis", region_name=cfg.aws_region)

    sent = 0
    for evt in build_events(args.count, args.fraud_ratio, args.invalid, args.attack):
        payload = to_json_safe(evt)
        partition_key = str(payload.get("user_or_card_key") or payload.get("TransactionID") or "unknown")
        label = payload.get("ground_truth_label", "?")
        if args.dry_run:
            print(f"[dry-run] {payload.get('TransactionID')} montant={payload.get('TransactionAmt')} "
                  f"carte={partition_key} label={label}")
        else:
            resp = kinesis.put_record(
                StreamName=stream,
                Data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                PartitionKey=partition_key,
            )
            print(f"[Kinesis] {payload.get('TransactionID')} -> shard {resp['ShardId']} "
                  f"(montant={payload.get('TransactionAmt')}, label={label})")
            time.sleep(args.delay)
        sent += 1

    print(f"\n{sent} événements {'affichés' if args.dry_run else f'envoyés vers {stream}'}.")


if __name__ == "__main__":
    main()
