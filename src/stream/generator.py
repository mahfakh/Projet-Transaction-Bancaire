"""Générateur et validateur de transactions temps réel (F1).

Fonctionnalités :
- Ingestion en continu ou par lots (rejeu du dataset ou génération synthétique)
- Validation stricte du format et rejet des événements corrompus dans une file DLQ
- Attribution d'un TransactionID unique et d'un horodatage ISO
- Support de scénarios d'attaque de démonstration (vélocité rapide, anomalie de montant, changement de terminal).
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.agent.schemas import TransactionEvent


class TransactionStreamGenerator:
    """Générateur de flux de transactions temps réel et validateur F1."""

    def __init__(
        self,
        dataset_path: str = "ieee-fraud-detection/train_transaction.csv",
        identity_path: str = "ieee-fraud-detection/train_identity.csv",
        dlq_output_path: str = "data/reports/dlq_rejected_transactions.json",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.identity_path = Path(identity_path)
        self.dlq_path = Path(dlq_output_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)
        self.rejected_events: List[Dict[str, Any]] = []

        # Chargement d'un réservoir d'exemples réels
        print("[Generator] Initialisation du réservoir de transactions...")
        df_tr = pd.read_csv(self.dataset_path, nrows=5000)
        df_id = pd.read_csv(self.identity_path, nrows=5000)
        self.pool = pd.merge(df_tr, df_id, on="TransactionID", how="left")
        print(f"[Generator] Réservoir prêt avec {len(self.pool)} transactions réelles.")

    def validate_event(self, raw_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Valide la conformité de l'événement entrant (F1)."""
        # Vérification des champs minimaux requis
        required_fields = ["TransactionAmt"]
        for field in required_fields:
            if field not in raw_data or raw_data[field] is None:
                return False, f"Champ obligatoire manquant : {field}"

        try:
            amt = float(raw_data["TransactionAmt"])
            if amt <= 0:
                return False, f"Montant invalide ou négatif : {amt}"
        except (ValueError, TypeError):
            return False, "Format de montant non numérique"

        return True, None

    def reject_event(self, raw_data: Dict[str, Any], reason: str) -> None:
        """Place l'événement invalide en file de rejet / Dead-Letter Queue (F1)."""
        record = {
            "rejected_at": datetime.now().isoformat(),
            "reason": reason,
            "raw_payload": raw_data,
        }
        self.rejected_events.append(record)
        # Écriture persistante
        with open(self.dlq_path, "w", encoding="utf-8") as f:
            json.dump(self.rejected_events, f, indent=2, ensure_ascii=False)

    def generate_event(
        self,
        scenario: str = "random",  # "random", "fraud_burst", "high_amount", "invalid_event"
    ) -> Optional[Dict[str, Any]]:
        """Génère un événement de transaction enrichi et validé."""
        # Scénario d'événement corrompu pour tester la file de rejet (F1)
        if scenario == "invalid_event":
            invalid_payload = {
                "TransactionID": f"INVALID_{int(time.time()*1000)}",
                "TransactionAmt": -50.0,  # Erreur volontaire
                "ProductCD": "UNKNOWN",
            }
            is_valid, reason = self.validate_event(invalid_payload)
            if not is_valid:
                self.reject_event(invalid_payload, reason or "Validation error")
                return None

        # Sélection d'une ligne de base du dataset
        if scenario == "fraud_burst":
            # Chercher une fraude avérée dans le réservoir
            frauds = self.pool[self.pool["isFraud"] == 1]
            row = frauds.sample(1).iloc[0].to_dict() if not frauds.empty else self.pool.sample(1).iloc[0].to_dict()
        else:
            row = self.pool.sample(1).iloc[0].to_dict()

        # Élimination stricte de la cible isFraud de la transaction ingérée (F2)
        true_label = int(row.pop("isFraud", 0))

        # Remplacement / enrichissement avec des données temps réel
        now_iso = datetime.now().isoformat()
        tx_id = f"TX_{int(time.time()*1000)}_{random.randint(100, 999)}"
        row["TransactionID"] = tx_id
        row["event_timestamp"] = now_iso
        row["ground_truth_label"] = true_label  # Conservé uniquement pour métriques/debug académique

        # Gestion du montant selon le scénario
        if scenario == "high_amount":
            row["TransactionAmt"] = round(random.uniform(1500.0, 5000.0), 2)

        # Clé synthétique de carte
        c1 = row.get("card1", 1000)
        c4 = row.get("card4", "visa")
        a1 = row.get("addr1", 100)
        row["user_or_card_key"] = f"USER_{c1}_{c4}_{a1}"

        # Validation
        is_valid, reason = self.validate_event(row)
        if not is_valid:
            self.reject_event(row, reason or "Validation failure")
            return None

        return row

    def stream_generator(
        self,
        batch_size: int = 10,
        delay_seconds: float = 0.5,
    ) -> Generator[Dict[str, Any], None, None]:
        """Génère un flux d'événements continus avec temporisation simulée."""
        for _ in range(batch_size):
            evt = self.generate_event()
            if evt:
                yield evt
            if delay_seconds > 0:
                time.sleep(delay_seconds)


if __name__ == "__main__":
    gen = TransactionStreamGenerator()
    print("\n[Stream Test] Test de génération d'un événement normal :")
    evt = gen.generate_event(scenario="random")
    print(f"ID: {evt['TransactionID']}, Montant: ${evt['TransactionAmt']}, Carte: {evt['card4']}")

    print("\n[Stream Test] Test de rejet d'événement corrompu (DLQ) :")
    gen.generate_event(scenario="invalid_event")
    print(f"Total événements rejetés dans DLQ : {len(gen.rejected_events)}")
