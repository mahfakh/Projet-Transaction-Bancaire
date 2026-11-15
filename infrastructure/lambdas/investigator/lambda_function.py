"""AWS Lambda 4: Agent LLM d'Investigation & Création d'Alerte (F5, F6).

Actions :
- Appel d'Amazon Bedrock (Claude 3 / Llama 3) avec prompt contraint et données pseudonymisées
- Génération d'une explication structurée (facteurs, manques, incertitude, recommandation)
- Création de l'alerte dans DynamoDB (FraudDetectionAlerts) avec statut PENDING
- Publication d'une notification Amazon SNS si risque HIGH.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any, Dict, List

import boto3

bedrock = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")

ALERTS_TABLE = os.environ.get("ALERTS_TABLE", "FraudDetectionAlerts")
table = dynamodb.Table(ALERTS_TABLE)

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

SYSTEM_PROMPT = """Tu es un Agent IA spécialisé dans l'investigation de signaux de fraude bancaire pour assister un analyste humain.
Règles impératives :
1. Tu ne prends AUCUNE décision de blocage ou de validation.
2. Tu ne dois JAMAIS inventer de données non fournies.
3. Tu ne dois JAMAIS modifier le score de risque calculé par le modèle ML.
4. Identifie les signaux observés et les informations manquantes.
5. Réponds UNIQUEMENT au format JSON strict avec les clés :
   - investigation_summary (string)
   - observed_risk_factors (list of strings)
   - missing_information (list of strings)
   - uncertainty (string : LOW, MODERATE ou HIGH)
   - suggested_review (string : recommandation pour l'analyste)
"""


def call_bedrock_llm(context: Dict[str, Any]) -> Dict[str, Any]:
    """Appelle Amazon Bedrock pour générer l'analyse structurée."""
    user_msg = f"Contexte de transaction suspecte :\n{json.dumps(context, ensure_ascii=False)}"
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_msg}"}],
    }

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(payload),
        )
        resp_data = json.loads(resp["body"].read().decode())
        raw_text = resp_data["content"][0]["text"]
        # Extraction JSON
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(raw_text[start:end])
        return json.loads(raw_text)
    except Exception as e:
        print(f"[BEDROCK FALLBACK] Erreur Bedrock ({e}), utilisation de la synthèse déterministe locale.")
        score = context.get("fraud_score_ml", 0.5)
        return {
            "investigation_summary": f"Alerte automatique : La transaction présente un score ML de {score:.2f} avec des écarts comportementaux significatifs.",
            "observed_risk_factors": context.get("model_contributing_factors", ["Score ML anormal"]),
            "missing_information": ["Authentification 3-D Secure directe", "Historique d'adresses antérieures"],
            "uncertainty": "MODERATE",
            "suggested_review": "Vérifier la légitimité auprès du porteur avant validation définitive.",
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    tx = event.get("transaction", {})
    signals = event.get("behavioral_signals", {})
    detection = event.get("detection_result", {})

    tx_id = str(tx.get("TransactionID", detection.get("transaction_id", "UNKNOWN")))
    alert_id = f"ALT_{tx_id}"
    score = float(detection.get("fraud_score", 0.0))
    risk_level = str(detection.get("risk_level", "INVESTIGATE"))

    # Préparation du contexte pseudonymisé pour le LLM (F5 & F9)
    llm_context = {
        "transaction_id": tx_id,
        "fraud_score_ml": score,
        "risk_level": risk_level,
        "amount": float(tx.get("TransactionAmt", tx.get("amount", 0.0))),
        "card_network": str(tx.get("card4", "visa")),
        "device_info": str(tx.get("DeviceInfo", "Inconnu")),
        "behavioral_context": signals,
        "model_contributing_factors": detection.get("top_contributing_factors", []),
    }

    # Appel de l'agent LLM
    analysis = call_bedrock_llm(llm_context)
    analysis["disclaimer"] = "Avis généré pour assistance humaine. Ne constitue pas un blocage bancaire automatique."

    # Création de l'alerte dans DynamoDB
    alert_item = {
        "alert_id": alert_id,
        "transaction_id": tx_id,
        "risk_level": risk_level,
        "fraud_score": Decimal(str(score)),
        "status": "PENDING",
        "created_at": tx.get("event_timestamp", ""),
        "updated_at": tx.get("event_timestamp", ""),
        "transaction_data": {
            "amount": Decimal(str(float(tx.get("TransactionAmt", tx.get("amount", 0.0))))),
            "card4": str(tx.get("card4", "visa")),
            "device_info": str(tx.get("DeviceInfo", "Inconnu")),
        },
        "detection_result": {
            "fraud_score": Decimal(str(score)),
            "risk_level": risk_level,
            "factors": detection.get("top_contributing_factors", []),
        },
        "llm_investigation": analysis,
    }

    try:
        table.put_item(Item=alert_item)
    except Exception as e:
        print(f"[DYNAMODB ERROR] Impossible d'écrire l'alerte {alert_id} : {e}")

    # Notification SNS si risque HIGH
    if risk_level == "HIGH" and SNS_TOPIC_ARN:
        try:
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"🚨 [URGENT] Alerte Fraude Majeure : {tx_id}",
                Message=f"Transaction: {tx_id}\nMontant: ${tx.get('TransactionAmt', 0)}\nScore: {score}\n\nRésumé LLM:\n{analysis.get('investigation_summary')}",
            )
        except Exception as e:
            print(f"[SNS ERROR] {e}")

    return {
        "statusCode": 200,
        "alert_id": alert_id,
        "alert": alert_item,
        "llm_investigation": analysis,
    }
