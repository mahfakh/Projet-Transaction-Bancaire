"""Tests de la Lambda de scoring (Kinesis), des étapes Step Functions et du producteur Kinesis.

Aucun appel AWS : Kinesis, SQS et Step Functions sont simulés par de faux objets.
"""

from __future__ import annotations

import base64
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pytest

from src.agent.schemas import BehavioralSignals, DetectionResult, RiskLevel
from src.aws import lambda_handler as lh
from src.aws.adapter import CloudStorageAdapter
from src.aws.orchestration import FraudDetectionPipeline
from src.models.scorer import FraudScorer
from src.stream.generator import TransactionStreamGenerator
from src.stream.kinesis_producer import build_events, to_json_safe
from src.stream.validation import validate_event


def _detection(score: float = 0.88, risk: RiskLevel = RiskLevel.HIGH) -> DetectionResult:
    return DetectionResult(
        transaction_id="TX_AWS_TEST",
        model_version="lgb_test",
        fraud_score=score,
        risk_level=risk,
        behavioral_signals=BehavioralSignals(
            transactions_last_24h=4, avg_amount_historical=100.0,
            amount_ratio_to_avg=15.0, is_amount_abnormal=True, device_changed=True,
        ),
    )


def _synthetic(fraudulent: bool, tx_id: str) -> dict:
    row = TransactionStreamGenerator._synthetic_row(fraudulent)
    row.pop("isFraud")
    return {**row, "TransactionID": tx_id}


class _FakeSQS:
    def __init__(self):
        self.messages = []

    def send_message(self, QueueUrl, MessageBody):
        self.messages.append(json.loads(MessageBody))


class _ExecutionAlreadyExists(Exception):
    pass


class _FakeSFN:
    class exceptions:
        ExecutionAlreadyExists = _ExecutionAlreadyExists

    def __init__(self):
        self.executions, self.duplicate = [], False

    def start_execution(self, stateMachineArn, name, input):
        if self.duplicate:
            raise _ExecutionAlreadyExists()
        self.executions.append({"name": name, "input": json.loads(input)})


def _kinesis_record(payload, seq, raw_data=None):
    data = raw_data if raw_data is not None else base64.b64encode(json.dumps(payload).encode()).decode()
    return {"kinesis": {"data": data, "sequenceNumber": str(seq), "partitionKey": "pk"}}


@pytest.fixture
def lambda_env(tmp_path, monkeypatch):
    """Lambda avec stockage local temporaire et faux clients SQS / Step Functions."""
    monkeypatch.setenv("DLQ_URL", "https://sqs.fake/dlq")
    monkeypatch.setenv("STATE_MACHINE_ARN", "arn:aws:states:us-east-1:000:stateMachine:test")
    pipeline = FraudDetectionPipeline(mode="local")
    pipeline.storage = CloudStorageAdapter(mode="local", storage_dir=str(tmp_path))
    fakes = {"sqs": _FakeSQS(), "stepfunctions": _FakeSFN()}
    monkeypatch.setattr(lh, "_pipeline", pipeline)
    monkeypatch.setattr(lh, "_storage", CloudStorageAdapter(mode="local", storage_dir=str(tmp_path)))
    monkeypatch.setattr(lh, "_client", lambda service: fakes[service])
    return pipeline, fakes


def test_score_handler_routes_valid_invalid_and_suspicious(lambda_env):
    pipeline, fakes = lambda_env
    random.seed(0)
    event = {"Records": [
        _kinesis_record(_synthetic(False, "TX_OK"), 1),
        _kinesis_record({"TransactionID": "TX_BAD", "TransactionAmt": -5}, 2),
        _kinesis_record(None, 3, raw_data="pas-du-base64-json"),
        _kinesis_record(_synthetic(True, "TX_FRAUD"), 4),
    ]}

    result = lh.score_handler(event)

    assert result == {"batchItemFailures": []}
    assert len(pipeline.storage.get_transactions()) == 2          # seules les valides sont scorées
    assert len(fakes["sqs"].messages) == 2                        # invalide + illisible -> DLQ
    assert "négatif" in fakes["sqs"].messages[0]["reason"]
    assert [e["name"] for e in fakes["stepfunctions"].executions] == ["TX_FRAUD"]
    sfn_input = fakes["stepfunctions"].executions[0]["input"]
    assert set(sfn_input) == {"transaction", "detection"}
    assert "C1" not in sfn_input["transaction"]                   # moindre privilège : champs utiles seulement


def test_score_handler_reports_failure_and_stops(lambda_env, monkeypatch):
    """Erreur technique : on renvoie la séquence en échec et on n'avance pas (ordre préservé)."""
    pipeline, _ = lambda_env
    calls = []

    def flaky(raw_tx):
        calls.append(raw_tx["TransactionID"])
        if raw_tx["TransactionID"] == "TX_2":
            raise RuntimeError("DynamoDB throttling simulé")
        return None, _detection(0.01, RiskLevel.LOW)

    monkeypatch.setattr(pipeline, "score_and_record", flaky)
    records = [_kinesis_record({"TransactionID": f"TX_{i}", "TransactionAmt": 10.0}, i) for i in (1, 2, 3)]
    assert lh.score_handler({"Records": records}) == {"batchItemFailures": [{"itemIdentifier": "2"}]}
    assert calls == ["TX_1", "TX_2"]


def test_duplicate_investigation_is_ignored(lambda_env):
    _, fakes = lambda_env
    fakes["stepfunctions"].duplicate = True
    lh._start_investigation({"TransactionID": "TX_DUP", "TransactionAmt": 10.0}, _detection())  # ne lève rien


def test_execution_name_is_valid():
    name = lh._execution_name("TX 123/abc:" + "x" * 100)
    assert len(name) == 80 and all(c.isalnum() or c in "_-" for c in name)


def test_step_functions_lambdas_roundtrip(lambda_env, capsys):
    """Simule Step Functions : Investigate -> CreateAlert, avec sérialisation JSON entre les étapes."""
    state = json.loads(json.dumps({
        "transaction": {"TransactionID": "TX_AWS_TEST", "TransactionAmt": 1500.0, "card4": "visa"},
        "detection": _detection(0.91).model_dump(mode="json"),
    }))
    state["investigation"] = json.loads(json.dumps(lh.investigate_handler(state)))
    capsys.readouterr()
    alert_out = lh.create_alert_handler(state)

    assert alert_out["alert_id"] == "ALT_TX_AWS_TEST" and alert_out["risk_level"] == "HIGH"
    assert "TX_AWS_TEST" in alert_out["notification"] and "aucun blocage" in alert_out["notification"]
    emf = next(json.loads(l) for l in capsys.readouterr().out.splitlines() if '"_aws"' in l)
    assert emf["HighRiskAlertCount"] == 1
    assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "BankingFraudDetection"
    assert lh._storage.get_alerts()[0]["llm_investigation"]["generated_by"] == "rule_based"

    # Investigation en échec (branche Catch) : l'alerte est quand même créée
    state.pop("investigation")
    assert "revue manuelle" in lh.create_alert_handler(state)["notification"]


def test_step_functions_definition_is_consistent():
    text = (BASE_DIR / "infrastructure" / "step_functions_definition.json").read_text(encoding="utf-8")
    definition = json.loads(text)
    states = definition["States"]
    assert definition["StartAt"] in states
    targets = [s.get("Next") for s in states.values()] + [
        c.get("Next") for s in states.values() for c in s.get("Choices", []) + s.get("Catch", [])]
    assert all(t in states for t in targets if t)
    for placeholder in ("${InvestigateFunctionArn}", "${CreateAlertFunctionArn}", "${AlertTopicArn}"):
        assert placeholder in text
    assert "123456789012" not in text  # plus de faux numéro de compte


def test_behavioral_escalation_rule():
    """Une rupture comportementale forte déclenche l'investigation sans modifier le score ML."""
    random.seed(2)
    scorer = FraudScorer()
    tx = _synthetic(False, "TX_ESC")
    calm = scorer.score_transaction(tx, BehavioralSignals())
    attack = scorer.score_transaction(tx, BehavioralSignals(
        transactions_last_24h=5, amount_ratio_to_avg=9.0, is_amount_abnormal=True, device_changed=True))
    assert calm.risk_level == RiskLevel.LOW
    assert attack.risk_level == RiskLevel.INVESTIGATE
    assert attack.fraud_score == calm.fraud_score  # le score ML n'est jamais modifié


def test_kinesis_producer_events():
    random.seed(1)
    events = list(build_events(count=6, fraud_ratio=0.5, invalid=2, attacks=1))
    assert len(events) == 6 + 2 + 5
    attack_keys = [e["user_or_card_key"] for e in events if str(e.get("user_or_card_key", "")).startswith("ATTACK_")]
    assert len(attack_keys) == 5 and len(set(attack_keys)) == 1   # même carte => même shard Kinesis
    assert sum(1 for e in events if not validate_event(e)[0]) == 2
    assert to_json_safe({"a": float("nan"), "b": np.int64(3)}) == {"a": None, "b": 3}
    json.dumps([to_json_safe(e) for e in events], allow_nan=False)  # JSON strictement standard
