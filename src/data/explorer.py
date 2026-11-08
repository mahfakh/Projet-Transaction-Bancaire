"""Module d'exploration des données IEEE-CIS (Étape 1 & F2).

Analyse la distribution des classes, les valeurs manquantes,
les jointures entre train_transaction et train_identity, et génère un rapport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class IEEEDataExplorer:
    """Explorateur du dataset IEEE-CIS Fraud Detection."""

    def __init__(self, raw_dir: str = "ieee-fraud-detection") -> None:
        self.raw_dir = Path(raw_dir)
        self.transaction_path = self.raw_dir / "train_transaction.csv"
        self.identity_path = self.raw_dir / "train_identity.csv"

    def run_exploration(self, sample_size: int = 100000) -> Dict[str, Any]:
        """Exécute une exploration statistique sur un échantillon représentatif."""
        print(f"[EDA] Chargement de l'échantillon ({sample_size} lignes)...")

        # Lecture de train_transaction
        df_tr = pd.read_csv(self.transaction_path, nrows=sample_size)
        total_rows_tr = sample_size
        cols_tr = list(df_tr.columns)

        # Calcul du déséquilibre de classe
        fraud_counts = df_tr["isFraud"].value_counts().to_dict()
        fraud_pct = float(df_tr["isFraud"].mean() * 100)

        # Lecture de train_identity
        df_id = pd.read_csv(self.identity_path, nrows=sample_size)
        cols_id = list(df_id.columns)

        # Vérification de la jointure sur TransactionID
        common_ids = set(df_tr["TransactionID"]).intersection(set(df_id["TransactionID"]))
        join_pct = float(len(common_ids) / len(df_tr) * 100)

        # Statistiques des montants
        amt_stats = {
            "mean": float(df_tr["TransactionAmt"].mean()),
            "std": float(df_tr["TransactionAmt"].std()),
            "min": float(df_tr["TransactionAmt"].min()),
            "median": float(df_tr["TransactionAmt"].median()),
            "max": float(df_tr["TransactionAmt"].max()),
            "fraud_mean_amt": float(df_tr[df_tr["isFraud"] == 1]["TransactionAmt"].mean()),
            "legit_mean_amt": float(df_tr[df_tr["isFraud"] == 0]["TransactionAmt"].mean()),
        }

        # Cartes les plus fréquentes
        top_cards = df_tr["card4"].value_counts(dropna=False).head(5).to_dict()
        top_products = df_tr["ProductCD"].value_counts(dropna=False).to_dict()

        # Synthèse des valeurs manquantes (colonnes avec le plus de nuls)
        null_series = (df_tr.isnull().sum() / len(df_tr) * 100).sort_values(ascending=False)
        top_missing_cols = {col: float(val) for col, val in null_series.head(10).items()}

        report: Dict[str, Any] = {
            "sample_size": sample_size,
            "train_transaction_columns": len(cols_tr),
            "train_identity_columns": len(cols_id),
            "class_distribution": {
                "legit_count": int(fraud_counts.get(0, 0)),
                "fraud_count": int(fraud_counts.get(1, 0)),
                "fraud_percentage": round(fraud_pct, 3),
            },
            "identity_join_rate_pct": round(join_pct, 2),
            "transaction_amt_statistics": amt_stats,
            "top_card_networks": {str(k): int(v) for k, v in top_cards.items()},
            "product_codes": {str(k): int(v) for k, v in top_products.items()},
            "top_missing_columns_pct": top_missing_cols,
        }

        return report

    def save_report(self, report: Dict[str, Any], output_path: str = "data/reports/eda_summary.json") -> None:
        """Sauvegarde le rapport au format JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[EDA] Rapport sauvegardé dans {output_path}")


if __name__ == "__main__":
    explorer = IEEEDataExplorer()
    summary = explorer.run_exploration(sample_size=100000)
    explorer.save_report(summary)
    print("\n[EDA] Résumé statistique :")
    print(f"- Total colonnes transaction : {summary['train_transaction_columns']}")
    print(f"- Taux de fraude observé : {summary['class_distribution']['fraud_percentage']}%")
    print(f"- Taux de couverture Identity : {summary['identity_join_rate_pct']}%")
    print(f"- Montant moyen fraude : ${summary['transaction_amt_statistics']['fraud_mean_amt']:.2f} vs légitime : ${summary['transaction_amt_statistics']['legit_mean_amt']:.2f}")
