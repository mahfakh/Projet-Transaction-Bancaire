"""AWS Lambda 2: Moteur de Contexte Comportemental & Historique (F4).

Actions :
- Interroge Amazon DynamoDB (Table FraudDetectionProfiles) pour récupérer l'historique du porteur
- Calcule les indicateurs comportementaux :
    * Vélocité des transactions sur 24h
    * Ratio du montant / moyenne historique
    * Détection d'anomalie statistique (> 2.5 sigma)
    * Changement d'équipement ou de localisation
- Met à jour l'historique glissant dans DynamoDB
- Transmet les signaux comportementaux au modèle de scoring (Lambda 3).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

import boto3

dynamodb = boto3.resource("dynamodb")
PROFILES_TABLE = os.environ.get("PROFILES_TABLE", "FraudDetectionProfiles")
table = dynamodb.Table(PROFILES_TABLE)


def decimal_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    # Extraction de la transaction (support direct ou sortie Step Functions)
    tx = event.get("transaction", event)
    profile_key = tx.get("user_or_card_key", f"CARD_{tx.get('card1', 1000)}_visa")
    amount = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
    device = str(tx.get("DeviceInfo", tx.get("device_info", "Unknown")))
    location = str(tx.get("addr1", tx.get("ip_or_location", "Unknown")))

    now = datetime.utcnow()
    now_iso = now.isoformat()

    # 1. Lecture de l'historique dans DynamoDB
    history = []
    try:
        resp = table.get_item(Key={"profile_key": profile_key})
        item = resp.get("Item", {})
        history = item.get("history", [])
    except Exception as e:
        print(f"[WARN] Lecture DynamoDB impossible ({e}), initialisation profil local.")

    # 2. Calcul des signaux comportementaux
    if not history:
        # Premier achat
        signals = {
            "transactions_last_24h": 1,
            "avg_amount_historical": round(amount, 2),
            "amount_ratio_to_avg": 1.0,
            "is_amount_abnormal": False,
            "device_changed": False,
            "location_changed": False,
            "synthetic_notes": ["Historique client synthétisé pour démonstration académique."],
        }
    else:
        # Vélocité 24h
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        txs_24h = [h for h in history if h.get("timestamp", "") >= cutoff_24h]
        velocity = len(txs_24h) + 1

        past_amts = [float(h["amount"]) for h in history]
        avg_amt = sum(past_amts) / len(past_amts)
        ratio = amount / max(avg_amt, 1.0)

        # Calcul écart-type
        variance = sum((x - avg_amt) ** 2 for x in past_amts) / len(past_amts)
        std_amt = max(variance ** 0.5, avg_amt * 0.25)
        is_abnormal = bool((amount > avg_amt + 2.5 * std_amt) and (ratio > 2.0))

        known_devices = {h.get("device") for h in history if h.get("device") != "Unknown"}
        dev_changed = bool(known_devices and device not in ["Unknown", "nan"] and device not in known_devices)

        known_locs = {h.get("location") for h in history if h.get("location") != "Unknown"}
        loc_changed = bool(known_locs and location not in ["Unknown", "nan"] and location not in known_locs)

        signals = {
            "transactions_last_24h": velocity,
            "avg_amount_historical": round(avg_amt, 2),
            "amount_ratio_to_avg": round(ratio, 2),
            "is_amount_abnormal": is_abnormal,
            "device_changed": dev_changed,
            "location_changed": loc_changed,
            "synthetic_notes": ["Historique client synthétisé pour démonstration académique."],
        }

    # 3. Mise à jour glissante de l'historique dans DynamoDB (limité à 30 transactions)
    new_entry = {
        "timestamp": now_iso,
        "amount": Decimal(str(amount)),
        "device": device,
        "location": location,
    }
    updated_history = [new_entry] + [
        {
            "timestamp": h["timestamp"],
            "amount": Decimal(str(h["amount"])),
            "device": h.get("device", "Unknown"),
            "location": h.get("location", "Unknown"),
        }
        for h in history[:29]
    ]

    try:
        table.put_item(
            Item={
                "profile_key": profile_key,
                "last_seen": now_iso,
                "history": updated_history,
            }
        )
    except Exception as e:
        print(f"[WARN] Écriture profil DynamoDB ({e})")

    tx["behavioral_signals"] = signals

    return {
        "statusCode": 200,
        "transaction": tx,
        "behavioral_signals": signals,
    }
