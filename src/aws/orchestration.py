"""Orchestrateur du pipeline de détection et d'investigation (Architecture D & Flux de traitement).

Le pipeline est découpé en deux phases réutilisables :

1. `score_and_record`     : moteur comportemental (F4) -> modèle ML (F3) -> persistance (F8)
2. `investigate_and_alert`: agent LLM (F5) -> alerte pour revue humaine (F6)

- En local, `process_transaction` enchaîne les deux phases dans le même processus.
- Sur AWS, la phase 1 tourne dans la Lambda de scoring (consommateur Kinesis) et la phase 2
  est orchestrée par AWS Step Functions (voir src/aws/lambda_handler.py et infrastructure/).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.llm_investigator import LLMInvestigator
from src.agent.schemas import (
    AlertReview,
    AlertStatus,
    BehavioralSignals,
    DetectionResult,
    LLMInvestigation,
    RiskLevel,
)
from src.aws.adapter import CloudStorageAdapter
from src.behavioral.context_engine import BehavioralContextEngine
from src.config import get_config

# Champs de la transaction transmis à la phase d'investigation (principe de moindre privilège :
# l'agent LLM et l'alerte ne reçoivent que ce qui leur est utile, pas les ~400 colonnes IEEE-CIS).
INVESTIGATION_TX_FIELDS = (
    "TransactionID", "TransactionAmt", "currency", "card4", "ProductCD",
    "DeviceInfo", "P_emaildomain", "user_or_card_key", "event_timestamp",
)


def get_transaction_id(raw_tx: Dict[str, Any]) -> str:
    return str(raw_tx.get("TransactionID", raw_tx.get("transaction_id", f"TX_{uuid.uuid4().hex[:8]}")))


def minimal_transaction(raw_tx: Dict[str, Any]) -> Dict[str, Any]:
    """Sous-ensemble pseudonymisé de la transaction, sérialisable en JSON."""
    out: Dict[str, Any] = {}
    for key in INVESTIGATION_TX_FIELDS:
        value = raw_tx.get(key)
        if value is None or (isinstance(value, float) and value != value):  # None ou NaN
            continue
        out[key] = value.item() if hasattr(value, "item") else value
    out["TransactionID"] = get_transaction_id(raw_tx)
    return out


def needs_investigation(detection: DetectionResult) -> bool:
    """Règle de seuillage : investigation si risque INVESTIGATE ou HIGH."""
    return detection.risk_level in (RiskLevel.INVESTIGATE, RiskLevel.HIGH)


def build_alert(
    raw_tx: Dict[str, Any],
    detection: DetectionResult,
    investigation: Optional[LLMInvestigation],
) -> AlertReview:
    """Construit l'alerte destinée à l'analyste humain (F6)."""
    tx_id = get_transaction_id(raw_tx)
    return AlertReview(
        alert_id=f"ALT_{tx_id}",
        transaction_id=tx_id,
        risk_level=detection.risk_level,
        fraud_score=detection.fraud_score,
        status=AlertStatus.PENDING,
        transaction_data={
            "amount": float(raw_tx.get("TransactionAmt", raw_tx.get("amount", 0.0))),
            "card4": str(raw_tx.get("card4", "visa")),
            "user_or_card_key": str(raw_tx.get("user_or_card_key", "")),
            "device_info": str(raw_tx.get("DeviceInfo", "")),
            "product_code": str(raw_tx.get("ProductCD", "")),
        },
        detection_result=detection.model_dump(mode="json"),
        llm_investigation=investigation,
    )


class FraudDetectionPipeline:
    """Pipeline central de détection orchestrant ML, Règles, Agent LLM et Stockage."""

    def __init__(
        self,
        config_path: str = str(BASE_DIR / "config" / "settings.yaml"),
        mode: Optional[str] = None,  # par défaut : configuration (APP_MODE)
    ) -> None:
        # Import local : le modèle (lightgbm, pandas...) n'est chargé que par les composants qui scorent
        from src.models.scorer import FraudScorer

        self.mode = mode or get_config().mode
        # Historique des porteurs : mémoire en local, table DynamoDB en live (persistant entre Lambdas)
        self.context_engine = BehavioralContextEngine.from_config(self.mode)
        self.scorer = FraudScorer(config_path=config_path)
        # Le fournisseur LLM est réglé indépendamment du mode de stockage (LLM_PROVIDER),
        # car le sandbox AWS Academy peut bloquer Bedrock même en mode "live".
        self.investigator = LLMInvestigator()
        self.storage = CloudStorageAdapter(mode=self.mode)

    # ---------- Phase 1 : scoring ----------
    def score_and_record(self, raw_tx: Dict[str, Any]) -> Tuple[BehavioralSignals, DetectionResult]:
        """Signaux comportementaux (F4) + score ML (F3) + enregistrement systématique (F8)."""
        signals = self.context_engine.analyze_and_update(raw_tx)
        detection = self.scorer.score_transaction(raw_tx, signals)
        self.storage.save_transaction(raw_tx, detection)
        return signals, detection

    # ---------- Phase 2 : investigation ----------
    def investigate_and_alert(
        self, raw_tx: Dict[str, Any], detection: DetectionResult
    ) -> Tuple[LLMInvestigation, AlertReview]:
        """Agent LLM (F5) puis création de l'alerte pour revue humaine (F6)."""
        investigation = self.investigator.investigate(raw_tx, detection)
        alert = build_alert(raw_tx, detection, investigation)
        self.storage.save_alert(alert)
        return investigation, alert

    def process_transaction(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le pipeline de bout en bout pour une transaction (mode local / démonstration)."""
        tx_id = get_transaction_id(raw_tx)
        signals, detection = self.score_and_record(raw_tx)

        investigation: Optional[LLMInvestigation] = None
        alert: Optional[AlertReview] = None
        if needs_investigation(detection):
            # L'agent ne reçoit que le contexte autorisé
            investigation, alert = self.investigate_and_alert(minimal_transaction(raw_tx), detection)

        return {
            "transaction_id": tx_id,
            "fraud_score": detection.fraud_score,
            "risk_level": detection.risk_level.value,
            "signals": signals.model_dump(),
            "has_alert": alert is not None,
            "alert_id": alert.alert_id if alert else None,
            "investigation": investigation.model_dump() if investigation else None,
        }

    def process_batch(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Traite un lot de transactions."""
        return [self.process_transaction(tx) for tx in transactions]


if __name__ == "__main__":
    pipeline = FraudDetectionPipeline()
    test_tx = {
        "TransactionID": "DEMO_1001",
        "TransactionAmt": 1250.0,
        "card1": 13926,
        "card4": "discover",
        "addr1": 315,
        "DeviceInfo": "Windows_Chrome_120",
        "P_emaildomain": "protonmail.com",
    }
    result = pipeline.process_transaction(test_tx)
    print("\n[Pipeline End-to-End Result]")
    print(f"Transaction ID : {result['transaction_id']}")
    print(f"Fraud Score    : {result['fraud_score']} ({result['risk_level']})")
    print(f"Alerte créée   : {result['has_alert']}")
    if result["investigation"]:
        print(f"LLM Résumé     : {result['investigation']['investigation_summary']}")
