"""Moteur d'analyse comportementale et contexte historique (F4).

Fonctionnalités :
- Maintien d'un historique glissant de transactions par profil (carte / utilisateur synthétique)
- Calcul d'indicateurs comportementaux explicables :
    - Vélocité des transactions (fréquence sur fenêtre glissante de 24h)
    - Écart aux habitudes de dépenses (ratio au montant moyen, anomalie > seuil sigma)
    - Détection de répétition rapprochée
    - Changement brutal d'équipement ou de localisation
- Documentation transparente des champs simulés (ne prétend pas être une identité bancaire réelle).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from src.agent.schemas import BehavioralSignals, TransactionEvent
from src.behavioral.history_store import InMemoryHistoryStore, build_history_store


def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Convertit un horodatage ISO en datetime « naïf » comparable.

    Les horodatages avec fuseau (ex. ...Z ou +01:00) sont ramenés en UTC puis le fuseau est
    retiré, pour éviter l'erreur Python de comparaison entre dates avec et sans fuseau.
    """
    if isinstance(value, datetime):
        dt = value
    elif value:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


class BehavioralContextEngine:
    """Moteur maintenant l'historique et évaluant les signaux comportementaux (F4)."""

    def __init__(
        self,
        max_history_per_card: int = 50,
        amount_sigma_threshold: float = 2.5,
        history_store: Any = None,
    ) -> None:
        self.max_history = max_history_per_card
        self.sigma_threshold = amount_sigma_threshold
        # Historique par porteur : mémoire en local, DynamoDB en live (voir history_store.py)
        self.store = history_store if history_store is not None else InMemoryHistoryStore()

    @classmethod
    def from_config(cls, mode: Optional[str] = None) -> "BehavioralContextEngine":
        """Construit le moteur avec le stockage adapté au mode (local / live)."""
        return cls(history_store=build_history_store(mode))

    def generate_profile_key(self, tx: Dict[str, Any]) -> str:
        """Génère une clé composite stable pour identifier le profil porteur."""
        card1 = str(tx.get("card1", "C1"))
        card4 = str(tx.get("card4", "visa")).lower()
        addr1 = str(tx.get("addr1", "A0"))
        user_key = tx.get("user_or_card_key")
        if user_key and str(user_key) not in ["None", ""]:
            return str(user_key)
        return f"CARD_{card1}_{card4}_{addr1}"

    def analyze_and_update(self, tx: Dict[str, Any]) -> BehavioralSignals:
        """Analyse la transaction courante par rapport à l'historique et met à jour ce dernier."""
        profile_key = self.generate_profile_key(tx)
        tx_id = str(tx.get("TransactionID", tx.get("transaction_id", "")))
        history = self.store.get(profile_key)

        # Idempotence : si cette transaction est déjà dans l'historique (nouvelle tentative
        # Kinesis après une erreur), on l'exclut du calcul et on ne la rajoute pas.
        already_recorded = bool(tx_id) and any(e.get("transaction_id") == tx_id for e in history)
        past_txs = [e for e in history if not (tx_id and e.get("transaction_id") == tx_id)]

        amount = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
        current_time = _parse_timestamp(tx.get("event_timestamp")) or datetime.now()

        device = str(tx.get("DeviceInfo", tx.get("device_info", "Unknown")))
        location = str(tx.get("addr1", tx.get("ip_or_location", "Unknown")))

        synthetic_notes: List[str] = [
            "L'historique client et les changements d'équipements sont synthétisés pour cette démonstration académique.",
            "Le dataset IEEE-CIS ne contient pas de données d'identité bancaire nominale réelle."
        ]

        if not past_txs:
            # Premier événement observé pour ce porteur
            signals = BehavioralSignals(
                transactions_last_24h=1,
                avg_amount_historical=amount,
                amount_ratio_to_avg=1.0,
                is_amount_abnormal=False,
                device_changed=False,
                location_changed=False,
                synthetic_notes=synthetic_notes,
            )
        else:
            # Calcul de la vélocité sur les dernières 24h
            cutoff_24h = current_time - timedelta(hours=24)
            txs_last_24h = [
                pt for pt in past_txs
                if (_parse_timestamp(pt.get("timestamp")) or current_time) >= cutoff_24h
            ]
            velocity = len(txs_last_24h) + 1  # Inclut la transaction en cours

            # Statistiques historiques sur les montants
            past_amounts = [float(pt["amount"]) for pt in past_txs]
            avg_historical = float(np.mean(past_amounts))
            std_historical = float(np.std(past_amounts)) if len(past_amounts) > 1 else max(15.0, avg_historical * 0.3)

            ratio = amount / max(avg_historical, 1.0)
            is_abnormal = (amount > avg_historical + self.sigma_threshold * std_historical) and (ratio > 2.0)

            # Vérification de changements d'environnement
            known_devices = {pt.get("device") for pt in past_txs if pt.get("device") and pt.get("device") != "Unknown"}
            device_changed = bool(known_devices and device not in ["Unknown", "nan", "None"] and device not in known_devices)

            known_locations = {pt.get("location") for pt in past_txs if pt.get("location") and pt.get("location") != "Unknown"}
            location_changed = bool(known_locations and location not in ["Unknown", "nan", "None"] and location not in known_locations)

            signals = BehavioralSignals(
                transactions_last_24h=velocity,
                avg_amount_historical=round(avg_historical, 2),
                amount_ratio_to_avg=round(ratio, 2),
                is_amount_abnormal=is_abnormal,
                device_changed=device_changed,
                location_changed=location_changed,
                synthetic_notes=synthetic_notes,
            )

        # Enregistrement dans l'historique glissant (N derniers événements), sauf doublon
        if not already_recorded:
            history.append({
                "transaction_id": tx_id,
                "amount": amount,
                "timestamp": current_time.isoformat(),  # chaîne ISO : stockable en JSON / DynamoDB
                "device": device,
                "location": location,
            })
            self.store.put(profile_key, history[-self.max_history:])

        return signals

    def get_history_summary(self, profile_key: str) -> Dict[str, Any]:
        """Retourne un résumé de l'historique pour l'agent LLM ou le dashboard."""
        txs = self.store.get(profile_key)
        if not txs:
            return {"total_transactions": 0, "status": "Nouveau porteur de carte"}
        amounts = [t["amount"] for t in txs]
        return {
            "total_transactions": len(txs),
            "historical_mean_amt": round(float(np.mean(amounts)), 2),
            "historical_max_amt": round(float(np.max(amounts)), 2),
            "distinct_devices": list({t["device"] for t in txs if t["device"] != "Unknown"}),
        }


if __name__ == "__main__":
    engine = BehavioralContextEngine()
    card_tx1 = {"card1": 1234, "card4": "mastercard", "addr1": 299, "TransactionAmt": 45.0, "DeviceInfo": "iPhone"}
    sig1 = engine.analyze_and_update(card_tx1)
    print("TX 1 signals:", sig1.model_dump())

    card_tx2 = {"card1": 1234, "card4": "mastercard", "addr1": 299, "TransactionAmt": 620.0, "DeviceInfo": "Linux_PC"}
    sig2 = engine.analyze_and_update(card_tx2)
    print("\nTX 2 signals (spike + device change):", sig2.model_dump())
