"""AWS Lambda 3: Service de Scoring ML & Seuillage de Risque (F3).

Supporte :
- Inférence locale via artefact LightGBM téléchargé depuis Amazon S3
- OU appel d'un endpoint Amazon SageMaker Serverless / Real-time
- Persistance systématique de la transaction scorée dans Amazon DynamoDB (FraudDetectionTransactions)
- Déclenchement conditionnel de l'investigation LLM si fraud_score >= seuil.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Dict, List

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sagemaker_runtime = boto3.client("sagemaker-runtime")

TRANSACTIONS_TABLE = os.environ.get("TRANSACTIONS_TABLE", "FraudDetectionTransactions")
table = dynamodb.Table(TRANSACTIONS_TABLE)

S3_BUCKET = os.environ.get("MODELS_S3_BUCKET", "")
MODEL_KEY = os.environ.get("MODEL_S3_KEY", "models/fraud_detector_model.joblib")
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "")

INVESTIGATION_THRESHOLD = float(os.environ.get("INVESTIGATION_THRESHOLD", "0.20"))
HIGH_RISK_THRESHOLD = float(os.environ.get("HIGH_RISK_THRESHOLD", "0.75"))

# Cache global du modèle pour warm start Lambda
_MODEL = None
_PREPROCESSOR = None
_FEATURE_ENGINEER = None


def load_model_from_s3() -> None:
    global _MODEL, _PREPROCESSOR, _FEATURE_ENGINEER
    if _MODEL is not None:
        return

    import joblib

    local_model_path = "/tmp/model.joblib"
    local_prep_path = "/tmp/prep.joblib"
    local_fe_path = "/tmp/fe.joblib"

    try:
        if S3_BUCKET:
            s3.download_file(S3_BUCKET, MODEL_KEY, local_model_path)
            s3.download_file(S3_BUCKET, "models/preprocessor.joblib", local_prep_path)
            try:
                s3.download_file(S3_BUCKET, "models/feature_engineer.joblib", local_fe_path)
                _FEATURE_ENGINEER = joblib.load(local_fe_path)
            except Exception:
                pass
            _MODEL = joblib.load(local_model_path)
            _PREPROCESSOR = joblib.load(local_prep_path)
    except Exception as e:
        print(f"[WARN] Impossible de charger le modèle depuis S3: {e}. Utilisation du scorer heuristique par défaut.")


def score_heuristic(tx: Dict[str, Any], signals: Dict[str, Any]) -> float:
    """Calcul heuristique de secours si SageMaker ou S3 sont inaccessibles."""
    score = 0.05
    amt = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
    if amt > 1000:
        score += 0.35
    elif amt > 400:
        score += 0.20

    if signals.get("is_amount_abnormal"):
        score += 0.30
    if signals.get("device_changed"):
        score += 0.15
    if signals.get("transactions_last_24h", 1) > 3:
        score += 0.15

    return min(round(score, 4), 0.99)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    tx = event.get("transaction", event)
    signals = event.get("behavioral_signals", tx.get("behavioral_signals", {}))
    tx_id = str(tx.get("TransactionID", "TX_UNKNOWN"))

    # Inférence ML
    score = None
    if SAGEMAKER_ENDPOINT:
        try:
            resp = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT,
                ContentType="application/json",
                Body=json.dumps(tx),
            )
            res_dict = json.loads(resp["Body"].read().decode())
            score = float(res_dict.get("fraud_score", 0.1))
        except Exception as e:
            print(f"[SAGEMAKER ERROR] {e}")

    if score is None:
        try:
            load_model_from_s3()
            if _MODEL and _PREPROCESSOR:
                import pandas as pd
                df_single = pd.DataFrame([tx])
                if _FEATURE_ENGINEER:
                    df_single = _FEATURE_ENGINEER.transform(df_single)
                X = _PREPROCESSOR.transform(df_single)
                score = float(_MODEL.predict_proba(X)[0, 1])
        except Exception as e:
            print(f"[MODEL ERROR] {e}")

    if score is None:
        score = score_heuristic(tx, signals)

    # Détermination du niveau de risque
    if score >= HIGH_RISK_THRESHOLD:
        risk_level = "HIGH"
    elif score >= INVESTIGATION_THRESHOLD:
        risk_level = "INVESTIGATE"
    else:
        risk_level = "LOW"

    # Facteurs d'explication
    factors: List[str] = []
    amt = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
    if amt > 500:
        factors.append(f"Montant élevé ({amt:.2f} USD)")
    if signals.get("is_amount_abnormal"):
        factors.append(f"Écart critique aux habitudes (Ratio {signals.get('amount_ratio_to_avg', 1.0):.1f}x)")
    if signals.get("device_changed"):
        factors.append(f"Nouveau terminal ({tx.get('DeviceInfo', 'Inconnu')})")
    if signals.get("transactions_last_24h", 1) > 2:
        factors.append(f"Vélocité anormale ({signals.get('transactions_last_24h')} tx / 24h)")

    detection_result = {
        "transaction_id": tx_id,
        "fraud_score": round(score, 4),
        "risk_level": risk_level,
        "model_version": "lgb_aws_v2",
        "top_contributing_factors": factors,
        "requires_investigation": bool(score >= INVESTIGATION_THRESHOLD),
    }

    # Sauvegarde dans DynamoDB FraudDetectionTransactions
    try:
        table.put_item(
            Item={
                "transaction_id": tx_id,
                "timestamp": tx.get("event_timestamp", ""),
                "amount": Decimal(str(amt)),
                "card_network": str(tx.get("card4", "visa")),
                "fraud_score": Decimal(str(round(score, 4))),
                "risk_level": risk_level,
                "factors": factors,
            }
        )
    except Exception as e:
        print(f"[WARN] Erreur enregistrement DynamoDB Transactions : {e}")

    return {
        "statusCode": 200,
        "transaction": tx,
        "behavioral_signals": signals,
        "detection_result": detection_result,
    }
