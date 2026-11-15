"""AWS Lambda 1: Validation et Ingestion des Transactions (F1).

Déclenché par : Amazon Kinesis Data Streams ou API Gateway.
Actions :
- Valide le format JSON et les champs obligatoires (TransactionAmt, card1, etc.)
- Normalise l'événement avec un UUID unique et un horodatage ISO
- Si invalide : Route l'événement vers la file Dead-Letter Queue (SQS DLQ)
- Si valide : Transmet l'événement à l'étape suivante (Step Functions / Lambda 2).
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import boto3

sqs = boto3.client("sqs")
DLQ_URL = os.environ.get("DLQ_URL", "")


def validate_payload(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Valide les critères minimaux de la transaction (F1)."""
    if "TransactionAmt" not in data and "amount" not in data:
        return False, "Champ montant manquant (TransactionAmt)"

    raw_amt = data.get("TransactionAmt", data.get("amount"))
    try:
        amt = float(raw_amt)
        if amt <= 0:
            return False, f"Montant négatif ou nul ({amt})"
    except (ValueError, TypeError):
        return False, "Format de montant non numérique"

    return True, "OK"


def send_to_dlq(payload: Dict[str, Any], reason: str) -> None:
    """Envoie l'événement rejeté dans Amazon SQS DLQ."""
    if not DLQ_URL:
        print(f"[DLQ LOCAL] Rejet : {reason} | Payload: {payload}")
        return

    record = {
        "rejected_at": datetime.utcnow().isoformat(),
        "reason": reason,
        "payload": payload,
    }
    try:
        sqs.send_message(
            QueueUrl=DLQ_URL,
            MessageBody=json.dumps(record, ensure_ascii=False),
        )
    except Exception as e:
        print(f"[ERROR] Impossible d'écrire dans la DLQ SQS : {e}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Point d'entrée AWS Lambda."""
    # Support Kinesis Records ou payload JSON direct (Step Functions / API Gateway)
    raw_records: List[Dict[str, Any]] = []

    if "Records" in event:
        for r in event["Records"]:
            if "kinesis" in r and "data" in r["kinesis"]:
                payload_str = base64.b64decode(r["kinesis"]["data"]).decode("utf-8")
                raw_records.append(json.loads(payload_str))
            elif "body" in r:
                raw_records.append(json.loads(r["body"]))
    else:
        # Événement unique ou direct
        raw_records.append(event)

    processed_events = []

    for tx in raw_records:
        is_valid, reason = validate_payload(tx)
        if not is_valid:
            send_to_dlq(tx, reason)
            continue

        # Normalisation
        tx_id = str(tx.get("TransactionID", f"TX_{uuid.uuid4().hex[:12].upper()}"))
        tx["TransactionID"] = tx_id
        tx["event_timestamp"] = tx.get("event_timestamp", datetime.utcnow().isoformat())
        tx["processed_at"] = datetime.utcnow().isoformat()

        # Clé synthétique de profil porteur
        c1 = str(tx.get("card1", "1000"))
        c4 = str(tx.get("card4", "visa")).lower()
        a1 = str(tx.get("addr1", "A0"))
        tx["user_or_card_key"] = tx.get("user_or_card_key", f"CARD_{c1}_{c4}_{a1}")

        processed_events.append(tx)

    return {
        "statusCode": 200,
        "valid_count": len(processed_events),
        "validated_events": processed_events,
        # Pour Step Functions (première transaction du lot)
        "transaction": processed_events[0] if processed_events else None,
    }
