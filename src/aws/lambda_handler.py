"""Points d'entrée AWS Lambda du système de détection de fraude.

Une seule image Docker, trois fonctions Lambda (le template choisit le handler) :

- `score_handler`        : consommateur Kinesis. Valide (F1), calcule les signaux (F4), score (F3),
                           enregistre (F8) puis, si le risque l'exige, démarre l'investigation
                           Step Functions. Les événements invalides partent en Dead-Letter Queue SQS.
- `investigate_handler`  : étape Step Functions « InvestigateTransaction » (agent LLM, F5).
- `create_alert_handler` : étape Step Functions « CreateAlert » (alerte pour revue humaine F6,
                           métrique CloudWatch F9, message de notification pour SNS).

Les objets coûteux (modèle ML, clients AWS) sont créés une seule fois par conteneur Lambda
et réutilisés entre les invocations (« warm start »).
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.schemas import DetectionResult, LLMInvestigation, RiskLevel
from src.aws.orchestration import build_alert, get_transaction_id, minimal_transaction, needs_investigation
from src.config import get_config
from src.stream.validation import validate_event

# Caches au niveau du module : survivent entre deux invocations d'un même conteneur
_pipeline = None
_investigator = None
_storage = None
_clients: Dict[str, Any] = {}


def _client(service: str) -> Any:
    if service not in _clients:
        import boto3
        _clients[service] = boto3.client(service, region_name=get_config().aws_region)
    return _clients[service]


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        from src.aws.orchestration import FraudDetectionPipeline
        _pipeline = FraudDetectionPipeline()
    return _pipeline


def _log(event_type: str, **fields: Any) -> None:
    """Log JSON structuré : facile à filtrer dans CloudWatch Logs Insights."""
    print(json.dumps({"event_type": event_type, **fields}, ensure_ascii=False, default=str))


# ======================================================================
# 1. Consommateur Kinesis
# ======================================================================

def _decode_record(record: Dict[str, Any]) -> Any:
    """Kinesis transmet les données encodées en base64."""
    return json.loads(base64.b64decode(record["kinesis"]["data"]))


def _send_to_dlq(payload: Any, reason: str, source: Dict[str, Any]) -> None:
    """Envoie un événement invalide dans la Dead-Letter Queue SQS, avec la raison du rejet (F1)."""
    body = {
        "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "source": source,
        "raw_payload": payload,
    }
    _client("sqs").send_message(
        QueueUrl=os.environ["DLQ_URL"],
        MessageBody=json.dumps(body, ensure_ascii=False, default=str),
    )
    _log("EVENT_REJECTED", reason=reason, **source)


def _execution_name(tx_id: str) -> str:
    """Nom d'exécution Step Functions : 80 caractères max, [A-Za-z0-9_-].

    Nom déterministe (dérivé de l'ID de transaction) : si Kinesis rejoue un lot,
    Step Functions refuse le doublon au lieu de lancer une seconde investigation.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", tx_id)[:80]


def _start_investigation(raw_tx: Dict[str, Any], detection: DetectionResult) -> None:
    sfn = _client("stepfunctions")
    try:
        sfn.start_execution(
            stateMachineArn=os.environ["STATE_MACHINE_ARN"],
            name=_execution_name(detection.transaction_id),
            input=json.dumps({
                "transaction": minimal_transaction(raw_tx),
                "detection": detection.model_dump(mode="json"),
            }, ensure_ascii=False, default=str),
        )
    except sfn.exceptions.ExecutionAlreadyExists:
        _log("INVESTIGATION_ALREADY_STARTED", transaction_id=detection.transaction_id)


def score_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Traite un lot d'enregistrements Kinesis.

    Retourne `batchItemFailures` (option ReportBatchItemFailures) : en cas d'erreur technique,
    Lambda reprend le lot à partir de l'enregistrement en échec, sans retraiter ceux qui précèdent.
    On s'arrête au premier échec pour préserver l'ordre des transactions d'un même porteur.
    """
    pipeline = _get_pipeline()
    stats = {"processed": 0, "rejected": 0, "investigations": 0}

    for record in event.get("Records", []):
        seq = record["kinesis"]["sequenceNumber"]
        source = {"sequence_number": seq, "partition_key": record["kinesis"].get("partitionKey")}
        try:
            try:
                raw_tx = _decode_record(record)
            except (ValueError, KeyError) as e:
                _send_to_dlq(record["kinesis"].get("data"), f"Enregistrement illisible : {e}", source)
                stats["rejected"] += 1
                continue

            is_valid, reason = validate_event(raw_tx)
            if not is_valid:
                _send_to_dlq(raw_tx, reason or "Validation échouée", source)
                stats["rejected"] += 1
                continue

            raw_tx.setdefault("TransactionID", get_transaction_id(raw_tx))
            _signals, detection = pipeline.score_and_record(raw_tx)
            stats["processed"] += 1
            _log("TRANSACTION_SCORED", transaction_id=detection.transaction_id,
                 fraud_score=detection.fraud_score, risk_level=detection.risk_level.value)

            if needs_investigation(detection):
                _start_investigation(raw_tx, detection)
                stats["investigations"] += 1

        except Exception as e:  # erreur technique (AWS indisponible...) => nouvelle tentative Kinesis
            _log("RECORD_FAILED", error=f"{type(e).__name__}: {e}", **source)
            _log("BATCH_SUMMARY", **stats, failed_at=seq)
            return {"batchItemFailures": [{"itemIdentifier": seq}]}

    _log("BATCH_SUMMARY", **stats)
    return {"batchItemFailures": []}


# ======================================================================
# 2. Étapes Step Functions
# ======================================================================

def investigate_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Étape « InvestigateTransaction » : produit l'analyse structurée de l'agent (F5)."""
    global _investigator
    if _investigator is None:
        from src.agent.llm_investigator import LLMInvestigator
        _investigator = LLMInvestigator()

    detection = DetectionResult.model_validate(event["detection"])
    investigation = _investigator.investigate(event["transaction"], detection)
    _log("INVESTIGATION_DONE", transaction_id=detection.transaction_id,
         generated_by=investigation.generated_by, uncertainty=investigation.uncertainty)
    return investigation.model_dump(mode="json")


def _emit_alert_metrics(risk_level: RiskLevel) -> None:
    """Publie les métriques CloudWatch au format EMF (Embedded Metric Format).

    CloudWatch extrait automatiquement les métriques de cette ligne de log : aucun appel API
    supplémentaire ni permission spécifique. `HighRiskAlertCount` alimente l'alarme du template.
    """
    metrics = {"AlertCount": 1, "HighRiskAlertCount": 1 if risk_level == RiskLevel.HIGH else 0}
    print(json.dumps({
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": get_config().metrics_namespace,
                "Dimensions": [[]],
                "Metrics": [{"Name": name, "Unit": "Count"} for name in metrics],
            }],
        },
        **metrics,
    }))


def _notification_text(alert: Any, investigation: Optional[LLMInvestigation]) -> str:
    lines = [
        f"Alerte {alert.risk_level.value} - transaction {alert.transaction_id}",
        f"Score de fraude ML : {alert.fraud_score:.2f}",
        f"Montant : {alert.transaction_data.get('amount', 0):.2f} USD",
    ]
    if investigation is not None:
        lines += [f"Synthèse : {investigation.investigation_summary}",
                  f"Recommandation : {investigation.suggested_review}"]
    else:
        lines.append("Investigation automatique indisponible : revue manuelle complète requise.")
    lines.append("Décision à prendre par un analyste dans le tableau de bord (aucun blocage automatique).")
    return "\n".join(lines)


def create_alert_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """Étape « CreateAlert » : enregistre l'alerte pour l'analyste (F6) et publie les métriques (F9).

    `investigation` est absent si l'étape précédente a échoué : l'alerte est quand même créée,
    pour qu'aucune transaction suspecte ne passe inaperçue.
    """
    global _storage
    if _storage is None:
        from src.aws.adapter import CloudStorageAdapter
        _storage = CloudStorageAdapter()

    detection = DetectionResult.model_validate(event["detection"])
    inv_data = event.get("investigation")
    investigation = LLMInvestigation.model_validate(inv_data) if inv_data else None

    alert = build_alert(event["transaction"], detection, investigation)
    _storage.save_alert(alert)
    _emit_alert_metrics(alert.risk_level)
    _log("ALERT_CREATED", alert_id=alert.alert_id, risk_level=alert.risk_level.value,
         with_investigation=investigation is not None)

    return {
        "alert_id": alert.alert_id,
        "risk_level": alert.risk_level.value,
        "notification": _notification_text(alert, investigation),
    }
