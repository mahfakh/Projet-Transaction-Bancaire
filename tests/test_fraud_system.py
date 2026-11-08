"""Suite de tests unitaires et d'intégration du système de détection de fraude (Section 7 & 10)."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from src.agent.llm_investigator import LLMInvestigator
from src.agent.schemas import (
    AlertReview,
    AlertStatus,
    BehavioralSignals,
    DetectionResult,
    RiskLevel,
)
from src.aws.adapter import CloudStorageAdapter
from src.aws.orchestration import FraudDetectionPipeline
from src.behavioral.context_engine import BehavioralContextEngine
from src.data.preprocessor import FraudDataPreprocessor
from src.models.scorer import FraudScorer
from src.stream.generator import TransactionStreamGenerator


def test_stream_generator_validation_and_dlq(tmp_path):
    """Test F1: Validation du format des événements et file de rejet DLQ."""
    dlq_file = tmp_path / "test_dlq.json"
    gen = TransactionStreamGenerator(dlq_output_path=str(dlq_file))

    # Événement valide
    valid_evt = {"TransactionAmt": 150.0, "card4": "visa"}
    is_valid, reason = gen.validate_event(valid_evt)
    assert is_valid is True
    assert reason is None

    # Événement invalide (montant négatif)
    invalid_evt = {"TransactionAmt": -20.0, "card4": "visa"}
    is_valid, reason = gen.validate_event(invalid_evt)
    assert is_valid is False
    assert "invalide ou négatif" in reason

    # Enregistrement en DLQ
    gen.reject_event(invalid_evt, reason)
    assert len(gen.rejected_events) == 1
    assert dlq_file.exists()


def test_preprocessor_no_target_leakage():
    """Test F2: Vérifie que isFraud ne fuit jamais dans les caractéristiques."""
    prep = FraudDataPreprocessor()
    assert "isFraud" not in prep.feature_names
    assert "isFraud" not in prep.numerical_cols
    assert "isFraud" not in prep.categorical_cols


def test_scorer_inference():
    """Test F3: Inférence du modèle et bornes du score."""
    scorer = FraudScorer()
    sample_tx = {
        "TransactionID": "TEST_TX_001",
        "TransactionAmt": 120.0,
        "ProductCD": "W",
        "card1": 10000,
        "card4": "visa",
    }
    signals = BehavioralSignals(
        transactions_last_24h=1,
        avg_amount_historical=120.0,
        amount_ratio_to_avg=1.0,
    )
    result = scorer.score_transaction(sample_tx, signals)

    assert 0.0 <= result.fraud_score <= 1.0
    assert result.risk_level in [RiskLevel.LOW, RiskLevel.INVESTIGATE, RiskLevel.HIGH]
    assert result.transaction_id == "TEST_TX_001"
    assert isinstance(result.top_contributing_factors, list)


def test_behavioral_context_anomaly_detection():
    """Test F4: Détection d'anomalie de montant et changement d'équipement."""
    engine = BehavioralContextEngine()
    tx_base = {
        "card1": 9999,
        "card4": "visa",
        "addr1": 200,
        "TransactionAmt": 50.0,
        "DeviceInfo": "Device_A",
    }

    # 1ère transaction (normale)
    sig1 = engine.analyze_and_update(tx_base)
    assert sig1.transactions_last_24h == 1
    assert sig1.is_amount_abnormal is False
    assert sig1.device_changed is False

    # 2ème transaction avec montant 20x supérieur et nouveau device
    tx_spike = {
        "card1": 9999,
        "card4": "visa",
        "addr1": 200,
        "TransactionAmt": 1000.0,
        "DeviceInfo": "Device_B",
    }
    sig2 = engine.analyze_and_update(tx_spike)
    assert sig2.transactions_last_24h == 2
    assert sig2.is_amount_abnormal is True
    assert sig2.device_changed is True
    assert len(sig2.synthetic_notes) > 0  # Mention explicite des données synthétisées


def test_llm_investigator_constraints_and_structure():
    """Test F5: Contraintes de l'Agent LLM (pas de décision finale, sortie structurée)."""
    investigator = LLMInvestigator(provider="mock")
    tx_data = {
        "TransactionAmt": 1500.0,
        "card4": "mastercard",
        "DeviceInfo": "Unknown_Device",
        "P_emaildomain": "anonymous.org",
    }
    detection = DetectionResult(
        transaction_id="TX_LLM_TEST",
        model_version="lgb_test",
        fraud_score=0.88,
        risk_level=RiskLevel.HIGH,
        behavioral_signals=BehavioralSignals(
            transactions_last_24h=4,
            avg_amount_historical=100.0,
            amount_ratio_to_avg=15.0,
            is_amount_abnormal=True,
            device_changed=True,
        ),
        top_contributing_factors=["Montant critique"],
    )

    inv = investigator.investigate(tx_data, detection)

    assert inv.transaction_id == "TX_LLM_TEST"
    assert len(inv.observed_risk_factors) > 0
    assert len(inv.missing_information) > 0
    assert inv.uncertainty in ["LOW", "MODERATE", "HIGH"]
    assert "Ne constitue pas un blocage automatique" in inv.disclaimer


def test_storage_and_analyst_decision_audit(tmp_path):
    """Test F6 & F8: Enregistrement et revue humaine avec journal d'audit."""
    adapter = CloudStorageAdapter(mode="local", storage_dir=str(tmp_path / "storage"))

    alert = AlertReview(
        alert_id="ALT_TEST_100",
        transaction_id="TX_100",
        risk_level=RiskLevel.HIGH,
        fraud_score=0.85,
        status=AlertStatus.PENDING,
    )
    adapter.save_alert(alert)

    # Récupération
    alerts = adapter.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["status"] == AlertStatus.PENDING.value

    # Revue par l'analyste
    ok = adapter.update_analyst_decision(
        alert_id="ALT_TEST_100",
        new_status=AlertStatus.CONFIRMED_FRAUD,
        analyst_decision="Fraude avérée après appel",
        analyst_note="Client a confirmé vol de carte",
    )
    assert ok is True

    # Vérification du nouveau statut
    updated = adapter.get_alerts()[0]
    assert updated["status"] == AlertStatus.CONFIRMED_FRAUD.value
    assert updated["analyst_decision"] == "Fraude avérée après appel"
    assert (tmp_path / "storage" / "audit_log.jsonl").exists()
