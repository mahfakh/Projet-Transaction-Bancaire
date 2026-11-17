"""Adaptateur de stockage et services cloud AWS / Mock local (F8, F9).

Offre une double implémentation transparente :
1. Mode "local" (Mock sécurisé sans frais) : persistance fichier/JSON dans `data/storage/`
2. Mode "live" (AWS Boto3) : DynamoDB

Points importants pour AWS :
- DynamoDB refuse les `float` Python : toutes les valeurs sont converties en `Decimal`
  à l'écriture (`to_dynamo`) puis reconverties en int/float à la lecture (`from_dynamo`).
- En mode "live", les erreurs AWS ne sont PAS masquées : elles remontent à l'appelant.
  Dans la Lambda, cela déclenche les nouvelles tentatives automatiques puis l'envoi
  en Dead-Letter Queue, au lieu de perdre silencieusement une transaction.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.schemas import AlertReview, AlertStatus, DetectionResult
from src.config import get_config


# ==========================
# CONVERSIONS DYNAMODB
# ==========================

def to_dynamo(value: Any) -> Any:
    """Convertit récursivement une structure Python en types acceptés par DynamoDB.

    - float -> Decimal (via str() pour éviter les artefacts binaires, ex: 0.1 -> 0.1000000000000000055)
    - NaN / infini -> None (DynamoDB ne sait pas les stocker)
    - types numpy -> types Python natifs
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return value
    if isinstance(value, dict):
        return {str(k): to_dynamo(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_dynamo(v) for v in value]
    if hasattr(value, "item") and callable(value.item):  # scalaires numpy (np.float32, np.int64...)
        value = value.item()
        if isinstance(value, bool):
            return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return Decimal(str(value))
    return str(value)


def from_dynamo(value: Any) -> Any:
    """Inverse de `to_dynamo` : Decimal -> int ou float."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {k: from_dynamo(v) for k, v in value.items()}
    if isinstance(value, list):
        return [from_dynamo(v) for v in value]
    return value


class CloudStorageAdapter:
    """Gestionnaire de persistance et d'alertes (DynamoDB ou Mock Local)."""

    def __init__(
        self,
        mode: Optional[str] = None,  # "local" ou "live" ; par défaut : configuration (APP_MODE)
        storage_dir: Optional[str] = None,
        aws_region: Optional[str] = None,
        alerts_table_name: Optional[str] = None,
        transactions_table_name: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.mode = mode or cfg.mode
        self.storage_dir = Path(storage_dir) if storage_dir else cfg.storage_dir
        self.aws_region = aws_region or cfg.aws_region
        self.alerts_table_name = alerts_table_name or cfg.alerts_table
        self.transactions_table_name = transactions_table_name or cfg.transactions_table

        if self.mode == "local":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.alerts_file = self.storage_dir / "alerts.json"
            self.transactions_file = self.storage_dir / "transactions.json"
            self.audit_log_file = self.storage_dir / "audit_log.jsonl"
            self._init_local_files()
        elif self.mode == "live":
            # Pas de repli silencieux vers le mode local : une erreur de configuration
            # AWS doit être visible immédiatement.
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=self.aws_region)
            self.alerts_table = dynamodb.Table(self.alerts_table_name)
            self.transactions_table = dynamodb.Table(self.transactions_table_name)
        else:
            raise ValueError(f"Mode de stockage inconnu : {self.mode!r}")

    def _init_local_files(self) -> None:
        """Initialise les fichiers de données locaux s'ils n'existent pas."""
        for path in (self.alerts_file, self.transactions_file):
            if not path.exists():
                self._write_json(path, [])

    @staticmethod
    def _read_json(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _scan_all(self, table: Any) -> List[Dict[str, Any]]:
        """Parcourt toute la table (DynamoDB pagine par blocs de 1 Mo)."""
        items: List[Dict[str, Any]] = []
        kwargs: Dict[str, Any] = {}
        while True:
            resp = table.scan(**kwargs)
            items.extend(resp.get("Items", []))
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return [from_dynamo(i) for i in items]

    # ==========================
    # TRANSACTIONS
    # ==========================

    def save_transaction(self, tx_dict: Dict[str, Any], detection: DetectionResult) -> None:
        """Enregistre une transaction et son score (F8)."""
        record = {
            "transaction_id": detection.transaction_id,
            "timestamp": str(tx_dict.get("event_timestamp") or datetime.now().isoformat()),
            "amount": float(tx_dict.get("TransactionAmt", tx_dict.get("amount", 0.0))),
            "card_network": str(tx_dict.get("card4", "visa")),
            "fraud_score": detection.fraud_score,
            "risk_level": detection.risk_level.value,
            "factors": detection.top_contributing_factors,
            "user_or_card_key": str(tx_dict.get("user_or_card_key", "")),
        }

        if self.mode == "local":
            all_txs = self._read_json(self.transactions_file)
            all_txs.insert(0, record)
            self._write_json(self.transactions_file, all_txs[:500])  # Limite à 500 pour légèreté
        else:
            self.transactions_table.put_item(Item=to_dynamo(record))

    def get_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les dernières transactions traitées (plus récentes d'abord)."""
        if self.mode == "local":
            return self._read_json(self.transactions_file)[:limit]
        txs = self._scan_all(self.transactions_table)
        txs.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
        return txs[:limit]

    # ==========================
    # ALERTES & REVUE HUMAINE (F6)
    # ==========================

    def save_alert(self, alert: AlertReview) -> None:
        """Enregistre ou met à jour une alerte pour revue analyste."""
        alert_dict = alert.model_dump(mode="json")
        alert_dict["updated_at"] = datetime.now().isoformat()

        if self.mode == "local":
            alerts = self._read_json(self.alerts_file)
            existing_idx = next((i for i, a in enumerate(alerts) if a["alert_id"] == alert.alert_id), None)
            if existing_idx is not None:
                alerts[existing_idx] = alert_dict
            else:
                alerts.insert(0, alert_dict)
            self._write_json(self.alerts_file, alerts)
        else:
            self.alerts_table.put_item(Item=to_dynamo(alert_dict))

    def get_alerts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère la file d'alertes (plus récentes d'abord), avec filtrage optionnel par statut."""
        if self.mode == "local":
            alerts = self._read_json(self.alerts_file)
        else:
            alerts = self._scan_all(self.alerts_table)
            alerts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
        if status and status != "ALL":
            alerts = [a for a in alerts if a.get("status") == status]
        return alerts

    def update_analyst_decision(
        self,
        alert_id: str,
        new_status: AlertStatus,
        analyst_decision: str,
        analyst_note: str,
    ) -> bool:
        """Enregistre la décision humaine et horodate l'action d'audit (F6, F8, F9).

        Retourne False si l'alerte n'existe pas.
        """
        timestamp = datetime.now().isoformat()

        if self.mode == "local":
            alerts = self._read_json(self.alerts_file)
            target = next((a for a in alerts if a["alert_id"] == alert_id), None)
            if target is None:
                return False
            target.update(
                status=new_status.value,
                analyst_decision=analyst_decision,
                analyst_note=analyst_note,
                updated_at=timestamp,
            )
            self._write_json(self.alerts_file, alerts)
        else:
            from botocore.exceptions import ClientError

            try:
                self.alerts_table.update_item(
                    Key={"alert_id": alert_id},
                    UpdateExpression="SET #s = :s, analyst_decision = :d, analyst_note = :n, updated_at = :u",
                    # Sans cette condition, DynamoDB créerait une alerte fantôme pour un ID inconnu
                    ConditionExpression="attribute_exists(alert_id)",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":s": new_status.value,
                        ":d": analyst_decision,
                        ":n": analyst_note,
                        ":u": timestamp,
                    },
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                    return False
                raise

        self._audit({
            "audit_timestamp": timestamp,
            "event_type": "ANALYST_REVIEW_DECISION",
            "alert_id": alert_id,
            "new_status": new_status.value,
            "decision": analyst_decision,
            "analyst_note": analyst_note,
        })
        return True

    def _audit(self, entry: Dict[str, Any]) -> None:
        """Journal d'audit (F9) : fichier JSONL en local, ligne JSON structurée en live.

        En live, la sortie standard est collectée automatiquement par CloudWatch Logs
        (Lambda) ; l'export dédié vers CloudWatch sera complété avec l'étape monitoring.
        """
        line = json.dumps(entry, ensure_ascii=False)
        if self.mode == "local":
            with open(self.audit_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            print(f"[AUDIT] {line}")


if __name__ == "__main__":
    adapter = CloudStorageAdapter()
    print(f"[CloudStorageAdapter] Initialisé en mode: {adapter.mode}")
    print(f"Alertes existantes : {len(adapter.get_alerts())}")
