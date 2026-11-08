"""Service de scoring et d'inférence temps réel (F3).

Fonctionnalités :
- Chargement du modèle sérialisé et du préprocesseur
- Évaluation unitaire ou par lot de transactions
- Attribution du score de fraude (0.0 à 1.0)
- Classification selon seuils configurables : LOW, INVESTIGATE, HIGH
- Extraction des principaux facteurs d'influence pour explicabilité.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.agent.schemas import DetectionResult, RiskLevel, TransactionEvent
from src.data.preprocessor import FraudDataPreprocessor


class FraudScorer:
    """Service d'inférence de détection de fraude."""

    def __init__(
        self,
        model_path: str = "data/models/fraud_detector_model.joblib",
        preprocessor_path: str = "data/models/preprocessor.joblib",
        config_path: str = "config/settings.yaml",
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        # Chargement de la configuration
        self.config = self._load_config(config_path)
        thresholds = self.config.get("detection", {}).get("thresholds", {})
        self.low_threshold: float = float(thresholds.get("low_risk", 0.20))
        self.investigate_threshold: float = float(thresholds.get("investigate", 0.50))
        self.high_threshold: float = float(thresholds.get("high_risk", 0.75))

        # Chargement des artefacts
        print(f"[Scorer] Chargement du modèle {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Chargement optionnel du Feature Engineer
        self.feature_engineer = None
        fe_path = Path("data/models/feature_engineer.joblib")
        if fe_path.exists():
            self.feature_engineer = joblib.load(fe_path)

        # Métadonnées
        self.model_version = "lgb_production_v1"
        metrics_file = Path("data/reports/model_metrics.json")
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.model_version = meta.get("model_version", self.model_version)

    def _load_config(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def determine_risk_level(self, score: float) -> RiskLevel:
        """Détermine le niveau de risque selon les seuils configurés (F3)."""
        if score >= self.high_threshold:
            return RiskLevel.HIGH
        elif score >= self.low_threshold:
            return RiskLevel.INVESTIGATE
        else:
            return RiskLevel.LOW

    def extract_top_contributing_factors(
        self,
        tx_row: pd.Series,
        score: float,
    ) -> List[str]:
        """Identifie les signaux saillants de la transaction pour l'explication."""
        factors: List[str] = []

        amt = float(tx_row.get("TransactionAmt", 0.0))
        if amt > 500:
            factors.append(f"Montant élevé ({amt:.2f} USD)")
        elif amt > 250:
            factors.append(f"Montant supérieur à la moyenne ({amt:.2f} USD)")

        # Équipement / IP
        dev = str(tx_row.get("DeviceInfo", ""))
        if dev and dev not in ["None", "nan", "Unknown"]:
            factors.append(f"Terminal identifié : {dev}")

        email = str(tx_row.get("P_emaildomain", ""))
        if email and email not in ["None", "nan", "gmail.com", "yahoo.com", "hotmail.com"]:
            factors.append(f"Domaine email non conventionnel ({email})")

        card_type = str(tx_row.get("card4", ""))
        if card_type and card_type not in ["None", "nan"]:
            factors.append(f"Réseau carte : {card_type}")

        dist = tx_row.get("dist1", None)
        if dist is not None and not np.isnan(dist) and float(dist) > 100:
            factors.append(f"Distance géographique estimée inhabituelle ({float(dist):.0f} km)")

        if score >= self.investigate_threshold and not factors:
            factors.append("Combinaison de scores de fréquence d'adresses et d'identifiants réseau anormale")

        return factors

    def score_transaction(
        self,
        tx_dict: Dict[str, Any],
        behavioral_signals: Any,
    ) -> DetectionResult:
        """Calcule le score ML et produit le contrat DetectionResult (F3)."""
        # Création d'un DataFrame d'une seule ligne
        df_single = pd.DataFrame([tx_dict])

        # Feature engineering si disponible
        if self.feature_engineer is not None:
            df_single = self.feature_engineer.transform(df_single)

        # Prétraitement
        X = self.preprocessor.transform(df_single)

        # Inférence de probabilité
        score = float(self.model.predict_proba(X)[0, 1])
        risk_level = self.determine_risk_level(score)

        # Facteurs contributifs
        factors = self.extract_top_contributing_factors(df_single.iloc[0], score)

        # Enrichissement avec les signaux comportementaux s'ils existent
        if hasattr(behavioral_signals, "is_amount_abnormal") and behavioral_signals.is_amount_abnormal:
            factors.insert(0, f"Écart critique aux habitudes de dépenses (Ratio {behavioral_signals.amount_ratio_to_avg:.1f}x)")
        if hasattr(behavioral_signals, "device_changed") and behavioral_signals.device_changed:
            factors.append("Alerte comportementale : Changement soudain d'équipement de connexion")

        tx_id = str(tx_dict.get("TransactionID", tx_dict.get("transaction_id", "UNKNOWN")))

        return DetectionResult(
            transaction_id=tx_id,
            model_version=self.model_version,
            fraud_score=round(score, 4),
            risk_level=risk_level,
            behavioral_signals=behavioral_signals,
            top_contributing_factors=factors,
        )


if __name__ == "__main__":
    scorer = FraudScorer()
    sample_tx = {
        "TransactionID": "3000001",
        "TransactionAmt": 350.50,
        "ProductCD": "W",
        "card1": 10000,
        "card4": "visa",
        "P_emaildomain": "anonymous.com",
    }
    from src.agent.schemas import BehavioralSignals
    dummy_signals = BehavioralSignals(
        transactions_last_24h=4,
        avg_amount_historical=80.0,
        amount_ratio_to_avg=4.38,
        is_amount_abnormal=True,
    )
    res = scorer.score_transaction(sample_tx, dummy_signals)
    print("\n[Scorer Test Result]")
    print(res.model_dump_json(indent=2))
