"""Adaptateur de stockage et services cloud AWS / Mock local (F8, F9).

Offre une double implémentation transparente :
1. Mode "local" (Mock sécurisé sans frais) : persistance fichier/JSON dans `data/storage/`
2. Mode "live" (AWS Boto3) : DynamoDB, S3, Kinesis, CloudWatch

Permet à l'application de fonctionner immédiatement en local pour évaluation académique,
tout en étant prête pour un déploiement AWS en production.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.schemas import AlertReview, AlertStatus, DetectionResult, RiskLevel


class CloudStorageAdapter:
    """Gestionnaire de persistance et d'alertes (DynamoDB/S3 ou Mock Local)."""

    def __init__(
        self,
        mode: str = "local",  # "local" ou "live"
        storage_dir: str = "data/storage",
        aws_region: str = "eu-west-3",
        alerts_table_name: str = "FraudDetectionAlerts",
        transactions_table_name: str = "FraudDetectionTransactions",
        s3_bucket: str = "fraud-detection-prototype-artifacts",
    ) -> None:
        self.mode = mode
        self.storage_dir = Path(storage_dir)
        self.aws_region = aws_region
        self.alerts_table_name = alerts_table_name
        self.transactions_table_name = transactions_table_name
        self.s3_bucket = s3_bucket

        # En mode local, création des dossiers de stockage
        if self.mode == "local":
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.alerts_file = self.storage_dir / "alerts.json"
            self.transactions_file = self.storage_dir / "transactions.json"
            self.audit_log_file = self.storage_dir / "audit_log.jsonl"
            self._init_local_files()
        else:
            self._init_aws_clients()

    def _init_local_files(self) -> None:
        """Initialise les fichiers de données locaux s'ils n'existent pas."""
        if not self.alerts_file.exists():
            with open(self.alerts_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.transactions_file.exists():
            with open(self.transactions_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _init_aws_clients(self) -> None:
        """Initialise les clients Boto3 pour le mode Live AWS."""
        try:
            import boto3
            self.dynamodb = boto3.resource("dynamodb", region_name=self.aws_region)
            self.s3 = boto3.client("s3", region_name=self.aws_region)
            self.cloudwatch = boto3.client("cloudwatch", region_name=self.aws_region)
            self.alerts_table = self.dynamodb.Table(self.alerts_table_name)
            self.transactions_table = self.dynamodb.Table(self.transactions_table_name)
        except Exception as e:
            print(f"[AWS Adapter] Erreur d'initialisation AWS Boto3 ({e}). Bascule en mode local.")
            self.mode = "local"
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.alerts_file = self.storage_dir / "alerts.json"
            self.transactions_file = self.storage_dir / "transactions.json"
            self.audit_log_file = self.storage_dir / "audit_log.jsonl"
            self._init_local_files()

    # ==========================
    # TRANSACTIONS
    # ==========================

    def save_transaction(self, tx_dict: Dict[str, Any], detection: DetectionResult) -> None:
        """Enregistre une transaction et son score (F8)."""
        record = {
            "transaction_id": detection.transaction_id,
            "timestamp": tx_dict.get("event_timestamp", datetime.now().isoformat()),
            "amount": float(tx_dict.get("TransactionAmt", tx_dict.get("amount", 0.0))),
            "card_network": str(tx_dict.get("card4", "visa")),
            "fraud_score": detection.fraud_score,
            "risk_level": detection.risk_level.value,
            "factors": detection.top_contributing_factors,
            "user_or_card_key": str(tx_dict.get("user_or_card_key", "")),
        }

        if self.mode == "local":
            with open(self.transactions_file, "r", encoding="utf-8") as f:
                all_txs = json.load(f)
            # Limite à 500 transactions en mémoire pour légèreté
            all_txs.insert(0, record)
            all_txs = all_txs[:500]
            with open(self.transactions_file, "w", encoding="utf-8") as f:
                json.dump(all_txs, f, indent=2, ensure_ascii=False)
        else:
            try:
                self.transactions_table.put_item(Item=record)
            except Exception as e:
                print(f"[AWS] Erreur put_item DynamoDB: {e}")

    def get_transactions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les dernières transactions traitées."""
        if self.mode == "local":
            with open(self.transactions_file, "r", encoding="utf-8") as f:
                txs = json.load(f)
            return txs[:limit]
        else:
            try:
                resp = self.transactions_table.scan(Limit=limit)
                return resp.get("Items", [])
            except Exception:
                return []

    # ==========================
    # ALERTES & REVUE HUMAINE (F6)
    # ==========================

    def save_alert(self, alert: AlertReview) -> None:
        """Enregistre ou met à jour une alerte pour revue analyste."""
        alert_dict = alert.model_dump()
        alert_dict["updated_at"] = datetime.now().isoformat()

        if self.mode == "local":
            with open(self.alerts_file, "r", encoding="utf-8") as f:
                alerts = json.load(f)

            # Mise à jour si existe déjà, sinon insertion en tête
            existing_idx = next((i for i, a in enumerate(alerts) if a["alert_id"] == alert.alert_id), None)
            if existing_idx is not None:
                alerts[existing_idx] = alert_dict
            else:
                alerts.insert(0, alert_dict)

            with open(self.alerts_file, "w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False)
        else:
            try:
                self.alerts_table.put_item(Item=alert_dict)
            except Exception as e:
                print(f"[AWS] Erreur enregistrement alerte DynamoDB: {e}")

    def get_alerts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Récupère la file d'alertes avec filtrage optionnel par statut."""
        if self.mode == "local":
            with open(self.alerts_file, "r", encoding="utf-8") as f:
                alerts = json.load(f)
            if status and status != "ALL":
                alerts = [a for a in alerts if a.get("status") == status]
            return alerts
        else:
            try:
                resp = self.alerts_table.scan()
                items = resp.get("Items", [])
                if status and status != "ALL":
                    items = [a for a in items if a.get("status") == status]
                return items
            except Exception:
                return []

    def update_analyst_decision(
        self,
        alert_id: str,
        new_status: AlertStatus,
        analyst_decision: str,
        analyst_note: str,
    ) -> bool:
        """Enregistre la décision humaine et horodate l'action d'audit (F6, F8, F9)."""
        timestamp = datetime.now().isoformat()

        if self.mode == "local":
            with open(self.alerts_file, "r", encoding="utf-8") as f:
                alerts = json.load(f)

            found = False
            for a in alerts:
                if a["alert_id"] == alert_id:
                    a["status"] = new_status.value
                    a["analyst_decision"] = analyst_decision
                    a["analyst_note"] = analyst_note
                    a["updated_at"] = timestamp
                    found = True
                    break

            if found:
                with open(self.alerts_file, "w", encoding="utf-8") as f:
                    json.dump(alerts, f, indent=2, ensure_ascii=False)

                # Journal d'audit CloudWatch / Audit Log (F9)
                audit_entry = {
                    "audit_timestamp": timestamp,
                    "event_type": "ANALYST_REVIEW_DECISION",
                    "alert_id": alert_id,
                    "new_status": new_status.value,
                    "decision": analyst_decision,
                    "analyst_note": analyst_note,
                }
                with open(self.audit_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
                return True
            return False
        else:
            try:
                self.alerts_table.update_item(
                    Key={"alert_id": alert_id},
                    UpdateExpression="SET #s = :s, analyst_decision = :d, analyst_note = :n, updated_at = :u",
                    ExpressionAttributeNames={"#s": "status"},
                    ExpressionAttributeValues={
                        ":s": new_status.value,
                        ":d": analyst_decision,
                        ":n": analyst_note,
                        ":u": timestamp,
                    },
                )
                return True
            except Exception as e:
                print(f"[AWS] Erreur update alerte: {e}")
                return False


if __name__ == "__main__":
    adapter = CloudStorageAdapter(mode="local")
    print(f"[CloudStorageAdapter] Initialisé en mode: {adapter.mode}")
    print(f"Alertes existantes : {len(adapter.get_alerts())}")
