"""AWS Lambda 5: API REST pour la Revue Analyste & Dashboard (F6, F7, F9).

Exposé via Amazon API Gateway :
- GET /alerts : Récupère la liste des alertes
- GET /transactions : Récupère les dernières transactions scorées
- POST /review : Consigne la décision d'un analyste et horodate l'audit dans CloudWatch
- POST /simulate : Ingestion d'une transaction de test en direct.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict

import boto3

dynamodb = boto3.resource("dynamodb")
ALERTS_TABLE = os.environ.get("ALERTS_TABLE", "FraudDetectionAlerts")
TRANSACTIONS_TABLE = os.environ.get("TRANSACTIONS_TABLE", "FraudDetectionTransactions")

alerts_tbl = dynamodb.Table(ALERTS_TABLE)
tx_tbl = dynamodb.Table(TRANSACTIONS_TABLE)


def decimal_serializer(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError


def build_response(status_code: int, body: Any) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        },
        "body": json.dumps(body, default=decimal_serializer),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    http_method = event.get("httpMethod", event.get("requestContext", {}).get("http", {}).get("method", "GET"))
    path = event.get("path", event.get("rawPath", "/"))

    # CORS preflight
    if http_method == "OPTIONS":
        return build_response(200, {"message": "OK"})

    try:
        # GET /alerts
        if http_method == "GET" and "alerts" in path:
            resp = alerts_tbl.scan(Limit=50)
            items = resp.get("Items", [])
            items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
            return build_response(200, items)

        # GET /transactions
        elif http_method == "GET" and "transactions" in path:
            resp = tx_tbl.scan(Limit=100)
            items = resp.get("Items", [])
            items.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)
            return build_response(200, items)

        # POST /review
        elif http_method == "POST" and "review" in path:
            body = json.loads(event.get("body", "{}"))
            alert_id = body.get("alert_id")
            new_status = body.get("new_status")
            analyst_decision = body.get("analyst_decision", "")
            analyst_note = body.get("analyst_note", "")
            now_iso = datetime.utcnow().isoformat()

            if not alert_id or not new_status:
                return build_response(400, {"error": "alert_id et new_status sont obligatoires"})

            # Mise à jour DynamoDB
            alerts_tbl.update_item(
                Key={"alert_id": alert_id},
                UpdateExpression="SET #s = :s, analyst_decision = :d, analyst_note = :n, updated_at = :u",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":s": new_status,
                    ":d": analyst_decision,
                    ":n": analyst_note,
                    ":u": now_iso,
                },
            )

            # Audit Trail dans CloudWatch Logs (F9)
            print(f"[AUDIT LOG] {now_iso} | Alerte: {alert_id} | Statut: {new_status} | Décision: {analyst_decision} | Note: {analyst_note}")

            return build_response(200, {"success": True, "alert_id": alert_id, "updated_at": now_iso})

        return build_response(404, {"error": f"Route non trouvée: {http_method} {path}"})

    except Exception as e:
        print(f"[API ERROR] {e}")
        return build_response(500, {"error": str(e)})
