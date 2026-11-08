"""Orchestrateur du pipeline de détection et d'investigation (Architecture D & Flux de traitement).

Coordonne le flux complet :
Générateur / Ingestion -> Validation (F1)
  -> Moteur comportemental (F4)
  -> Modèle ML (F3)
  -> Règle de seuillage
  -> Agent LLM d'investigation (F5) si seuil dépassé
  -> Persistance DynamoDB / S3 / Local (F8)
  -> Notification & Supervision (F6, F9)
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
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
from src.models.scorer import FraudScorer


class FraudDetectionPipeline:
    """Pipeline central de détection orchestrant ML, Règles, Agent LLM et Stockage."""

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        mode: str = "local",
    ) -> None:
        self.mode = mode
        self.context_engine = BehavioralContextEngine()
        self.scorer = FraudScorer(config_path=config_path)
        self.investigator = LLMInvestigator(provider="mock" if mode == "local" else "bedrock")
        self.storage = CloudStorageAdapter(mode=mode)

    def process_transaction(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le pipeline de bout en bout pour une transaction."""
        tx_id = str(raw_tx.get("TransactionID", raw_tx.get("transaction_id", f"TX_{uuid.uuid4().hex[:8]}")))

        # 1. Analyse comportementale et contexte historique (F4)
        signals: BehavioralSignals = self.context_engine.analyze_and_update(raw_tx)

        # 2. Détection et scoring ML (F3)
        detection: DetectionResult = self.scorer.score_transaction(raw_tx, signals)

        # 3. Enregistrement systématique de la transaction (F8)
        self.storage.save_transaction(raw_tx, detection)

        investigation: Optional[LLMInvestigation] = None
        alert: Optional[AlertReview] = None

        # 4. Déclenchement de l'Agent LLM si investigation requise (F5, F6)
        # Seuil : risk_level != LOW (soit INVESTIGATE ou HIGH)
        if detection.risk_level in [RiskLevel.INVESTIGATE, RiskLevel.HIGH]:
            # L'agent ne reçoit que le contexte autorisé
            investigation = self.investigator.investigate(raw_tx, detection)

            # Création de l'alerte pour l'analyste humain (F6)
            alert = AlertReview(
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
                detection_result=detection.model_dump(),
                llm_investigation=investigation,
            )
            self.storage.save_alert(alert)

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
    pipeline = FraudDetectionPipeline(mode="local")
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
