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


# ==========================================================
# Tests de préparation au déploiement AWS (sans appel réseau)
# ==========================================================

from decimal import Decimal

import numpy as np

from src.agent.llm_investigator import extract_json_object
from src.aws.adapter import from_dynamo, to_dynamo
from src.config import get_config


def _sample_detection(score: float = 0.88) -> DetectionResult:
    return DetectionResult(
        transaction_id="TX_AWS_TEST",
        model_version="lgb_test",
        fraud_score=score,
        risk_level=RiskLevel.HIGH,
        behavioral_signals=BehavioralSignals(
            transactions_last_24h=4, avg_amount_historical=100.0,
            amount_ratio_to_avg=15.0, is_amount_abnormal=True, device_changed=True,
        ),
    )


def test_dynamo_conversion_roundtrip():
    """Fix 1 : DynamoDB refuse les float -> conversion Decimal aller-retour sans perte."""
    data = {
        "score": 0.8731, "amount": 125.5, "count": 3, "flag": True, "none": None,
        "nan": float("nan"), "np_float": np.float32(0.25), "np_int": np.int64(7),
        "nested": {"ratios": [1.5, 2.0], "label": "HIGH"},
    }
    converted = to_dynamo(data)
    assert converted["score"] == Decimal("0.8731")  # pas d'artefact binaire
    assert isinstance(converted["np_float"], Decimal)
    assert converted["np_int"] == 7 and converted["flag"] is True
    assert converted["nan"] is None  # DynamoDB ne stocke pas NaN

    def no_float(v):
        if isinstance(v, dict):
            return all(no_float(x) for x in v.values())
        if isinstance(v, list):
            return all(no_float(x) for x in v)
        return not isinstance(v, float)
    assert no_float(converted)

    back = from_dynamo(converted)
    assert back["score"] == 0.8731 and back["count"] == 3
    assert back["nested"]["ratios"] == [1.5, 2]


class _FakeTable:
    """Imite une table DynamoDB boto3 en refusant les float comme le vrai service."""

    def __init__(self, fail: bool = False):
        self.items, self.fail = {}, fail

    def put_item(self, Item):
        if self.fail:
            raise RuntimeError("AccessDenied simulé")

        def check(v):
            if isinstance(v, float):
                raise TypeError("Float types are not supported. Use Decimal types instead.")
            for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else []):
                check(x)
        check(Item)
        key = Item.get("alert_id") or Item.get("transaction_id")
        self.items[key] = Item

    def scan(self, **kwargs):
        return {"Items": list(self.items.values())}


def _live_adapter(fail: bool = False) -> CloudStorageAdapter:
    adapter = CloudStorageAdapter.__new__(CloudStorageAdapter)
    adapter.mode = "live"
    adapter.alerts_table = _FakeTable(fail)
    adapter.transactions_table = _FakeTable(fail)
    return adapter


def test_live_adapter_writes_decimal_and_reads_back():
    """Fix 1 : en mode live, les écritures passent la contrainte Decimal de DynamoDB."""
    adapter = _live_adapter()
    detection = _sample_detection()
    adapter.save_transaction({"TransactionAmt": 150.75, "card4": "visa"}, detection)
    adapter.save_alert(AlertReview(
        alert_id="ALT_1", transaction_id="TX_AWS_TEST", risk_level=RiskLevel.HIGH,
        fraud_score=0.88, detection_result=detection.model_dump(),
    ))
    assert adapter.get_transactions()[0]["amount"] == 150.75
    alert = adapter.get_alerts()[0]
    assert alert["fraud_score"] == 0.88 and alert["risk_level"] == "HIGH"


def test_live_adapter_does_not_swallow_errors():
    """Fix 6 : une erreur AWS doit remonter (=> retry Lambda puis DLQ), pas être juste affichée."""
    adapter = _live_adapter(fail=True)
    with pytest.raises(RuntimeError):
        adapter.save_transaction({"TransactionAmt": 10.0}, _sample_detection())


def test_config_env_overrides(monkeypatch):
    """Fix 3 : les variables d'environnement (injectées par le template AWS) priment sur settings.yaml."""
    monkeypatch.setenv("APP_MODE", "live")
    monkeypatch.setenv("ALERTS_TABLE", "FraudDetectionAlerts-prototype")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    get_config.cache_clear()
    try:
        cfg = get_config()
        assert cfg.is_live
        assert cfg.alerts_table == "FraudDetectionAlerts-prototype"
        assert cfg.aws_region == "us-east-1"
    finally:
        get_config.cache_clear()

    monkeypatch.setenv("APP_MODE", "production")
    with pytest.raises(ValueError):
        get_config()
    get_config.cache_clear()


def test_extract_json_from_llm_reply():
    """Fix 8 : le JSON est extrait même si le modèle ajoute du texte ou des balises ```json."""
    reply = 'Voici mon analyse :\n```json\n{"investigation_summary": "x", "suggested_review": "y"}\n```\nCordialement.'
    assert extract_json_object(reply)["suggested_review"] == "y"
    with pytest.raises(ValueError):
        extract_json_object("pas de JSON ici")


class _FakeBedrock:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def converse(self, **kwargs):
        if self.error:
            raise self.error
        return {"output": {"message": {"content": [{"text": self.reply}]}}}


def test_bedrock_success_is_labelled():
    """Fix 8 : une réponse Bedrock valide est utilisée et étiquetée avec le modèle."""
    agent = LLMInvestigator(provider="bedrock", model_id="test-model")
    agent._bedrock_client = _FakeBedrock(reply=(
        'Analyse : {"investigation_summary": "Montant 15x supérieur", "observed_risk_factors": ["montant"],'
        ' "missing_information": ["3DS"], "uncertainty": "low", "suggested_review": "Appeler le client"}'
    ))
    inv = agent.investigate({"TransactionAmt": 1500.0}, _sample_detection())
    assert inv.generated_by == "bedrock:test-model"
    assert inv.uncertainty == "LOW"
    assert inv.transaction_id == "TX_AWS_TEST"


def test_bedrock_failure_falls_back_and_says_so():
    """Fix 8 : si Bedrock est bloqué (cas AWS Academy), repli déterministe explicitement signalé."""
    agent = LLMInvestigator(provider="bedrock", model_id="test-model")
    agent._bedrock_client = _FakeBedrock(error=RuntimeError("AccessDeniedException"))
    inv = agent.investigate({"TransactionAmt": 1500.0}, _sample_detection())
    assert inv.generated_by == "rule_based_fallback"
    assert len(inv.observed_risk_factors) > 0

    # Réponse incomplète (pas de résumé) => rejetée puis repli
    agent._bedrock_client = _FakeBedrock(reply='{"uncertainty": "LOW"}')
    assert agent.investigate({"TransactionAmt": 1.0}, _sample_detection()).generated_by == "rule_based_fallback"


def test_synthetic_generator_without_dataset(tmp_path):
    """Le générateur fonctionne sans le dataset IEEE-CIS (nécessaire pour alimenter Kinesis)."""
    gen = TransactionStreamGenerator(dataset_path=str(tmp_path / "absent.csv"), dlq_output_path=str(tmp_path / "dlq.json"))
    evt = gen.generate_event(scenario="fraud_burst")
    assert evt is not None and evt["TransactionAmt"] > 0
    assert "isFraud" not in evt  # la cible ne fuit jamais dans l'événement ingéré
    assert gen.generate_event(scenario="invalid_event") is None
    assert len(gen.rejected_events) == 1


# ==========================================================
# Fix 2 : historique des porteurs persistant (DynamoDB)
# ==========================================================

from src.behavioral.history_store import DynamoHistoryStore


class _FakeHistoryTable:
    """Imite la table DynamoDB d'historique (refuse les float comme le vrai service)."""

    def __init__(self):
        self.items = {}

    def get_item(self, Key, ConsistentRead=False):
        item = self.items.get(Key["profile_key"])
        return {"Item": item} if item else {}

    def put_item(self, Item):
        def check(v):
            if isinstance(v, float):
                raise TypeError("Float types are not supported. Use Decimal types instead.")
            for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else []):
                check(x)
        check(Item)
        assert isinstance(Item["expires_at"], int)  # attribut TTL au format epoch
        self.items[Item["profile_key"]] = Item


def _tx(tx_id, amount, device, ts):
    return {"TransactionID": tx_id, "user_or_card_key": "USER_42", "TransactionAmt": amount,
            "DeviceInfo": device, "addr1": 299, "event_timestamp": ts}


def test_history_survives_across_lambda_invocations():
    """Deux moteurs distincts (= deux exécutions Lambda) partagent l'historique via DynamoDB."""
    table = _FakeHistoryTable()

    first_lambda = BehavioralContextEngine(history_store=DynamoHistoryStore("t", "us-east-1", table=table))
    for i, amt in enumerate([40.0, 55.0, 48.0]):
        first_lambda.analyze_and_update(_tx(f"TX_{i}", amt, "iPhone", f"2026-11-17T10:0{i}:00"))

    # Nouvelle instance : mémoire vide, seule la table relie les deux exécutions
    second_lambda = BehavioralContextEngine(history_store=DynamoHistoryStore("t", "us-east-1", table=table))
    sig = second_lambda.analyze_and_update(_tx("TX_SPIKE", 1200.0, "Linux_PC", "2026-11-17T10:05:00"))

    assert sig.transactions_last_24h == 4
    assert sig.is_amount_abnormal is True
    assert sig.device_changed is True
    assert len(table.items["USER_42"]["events"]) == 4


def test_history_is_idempotent_on_retry():
    """Une nouvelle tentative Kinesis (même TransactionID) ne compte pas la transaction deux fois."""
    engine = BehavioralContextEngine()
    engine.analyze_and_update(_tx("TX_A", 50.0, "iPhone", "2026-11-17T10:00:00"))
    first = engine.analyze_and_update(_tx("TX_B", 60.0, "iPhone", "2026-11-17T10:01:00"))
    retry = engine.analyze_and_update(_tx("TX_B", 60.0, "iPhone", "2026-11-17T10:01:00"))

    assert first.transactions_last_24h == retry.transactions_last_24h == 2
    assert len(engine.store.get("USER_42")) == 2


def test_history_window_and_cap():
    """Fenêtre de 24h respectée, fuseaux mélangés acceptés, historique plafonné."""
    engine = BehavioralContextEngine(max_history_per_card=3)
    engine.analyze_and_update(_tx("OLD", 50.0, "iPhone", "2026-11-15T09:00:00Z"))  # avec fuseau, > 24h
    for i in range(3):
        sig = engine.analyze_and_update(_tx(f"NEW_{i}", 50.0, "iPhone", f"2026-11-17T10:0{i}:00"))
    assert sig.transactions_last_24h == 3  # l'ancienne transaction est hors fenêtre
    assert [e["transaction_id"] for e in engine.store.get("USER_42")] == ["NEW_0", "NEW_1", "NEW_2"]
