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
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np

from src.agent.schemas import BehavioralSignals, TransactionEvent


class BehavioralContextEngine:
    """Moteur maintenant l'historique et évaluant les signaux comportementaux (F4)."""

    def __init__(
        self,
        max_history_per_card: int = 50,
        amount_sigma_threshold: float = 2.5,
    ) -> None:
        self.max_history = max_history_per_card
        self.sigma_threshold = amount_sigma_threshold
        # Stockage en mémoire : clé profil -> liste d'événements passés
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

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
        past_txs = self._history[profile_key]

        amount = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
        current_time_str = tx.get("event_timestamp")
        try:
            current_time = datetime.fromisoformat(current_time_str) if current_time_str else datetime.now()
        except Exception:
            current_time = datetime.now()

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
                if pt.get("timestamp", current_time) >= cutoff_24h
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

        # Enregistrement dans l'historique glissant
        self._history[profile_key].append({
            "amount": amount,
            "timestamp": current_time,
            "device": device,
            "location": location,
        })
        if len(self._history[profile_key]) > self.max_history:
            self._history[profile_key].pop(0)

        return signals

    def get_history_summary(self, profile_key: str) -> Dict[str, Any]:
        """Retourne un résumé de l'historique pour l'agent LLM ou le dashboard."""
        txs = self._history.get(profile_key, [])
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
