"""Module de préparation et d'enrichissement des données (F2).

Gère :
- La jointure TransactionID entre transaction et identity
- Le nettoyage des valeurs manquantes et types
- L'encodage des variables catégorielles
- La prévention stricte des fuites de données (isFraud exclu)
- L'enregistrement des métadonnées du préprocesseur pour reproductibilité en inférence temps réel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# Sous-ensemble robuste et représentatif des caractéristiques IEEE-CIS
NUMERICAL_COLS = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "C1",
    "C2",
    "C4",
    "C6",
    "C7",
    "C8",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D15",
    "V12",
    "V13",
    "V14",
    "V15",
    "V29",
    "V30",
    "V75",
    "V76",
    "V79",
    "V80",
    "V90",
    "V91",
    "V126",
    "V127",
    "V128",
    "V130",
    "V131",
    "V281",
    "V282",
    "V283",
    "V306",
    "V307",
    "V308",
    "V310",
    "V312",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


class FraudDataPreprocessor:
    """Pipeline de prétraitement pour l'entraînement et l'inférence temps réel."""

    def __init__(self) -> None:
        self.numerical_cols: List[str] = NUMERICAL_COLS.copy()
        self.categorical_cols: List[str] = CATEGORICAL_COLS.copy()
        self.feature_names: List[str] = self.numerical_cols + self.categorical_cols
        self.category_mappings: Dict[str, Dict[str, int]] = {}
        self.numerical_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def merge_datasets(
        self,
        df_transaction: pd.DataFrame,
        df_identity: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Jointure sécurisée sur TransactionID (F2)."""
        if df_identity is not None and not df_identity.empty:
            merged = pd.merge(df_transaction, df_identity, on="TransactionID", how="left")
        else:
            merged = df_transaction.copy()
        return merged

    def fit(self, df: pd.DataFrame) -> FraudDataPreprocessor:
        """Calcule les médianes et dictionnaires d'encodage (sans la cible isFraud)."""
        # Calcul des médianes numériques
        for col in self.numerical_cols:
            if col in df.columns:
                val = float(df[col].median(skipna=True))
                self.numerical_medians[col] = 0.0 if np.isnan(val) else val
            else:
                self.numerical_medians[col] = 0.0

        # Construction des mappings catégoriels (label encoding avec 0 réservé aux inconnus)
        for col in self.categorical_cols:
            if col in df.columns:
                unique_vals = (
                    df[col].dropna().astype(str).value_counts().index.tolist()
                )
                # On limite aux 150 catégories les plus fréquentes pour éviter l'explosion
                top_cats = unique_vals[:150]
                mapping = {cat: idx + 1 for idx, cat in enumerate(top_cats)}
                self.category_mappings[col] = mapping
            else:
                self.category_mappings[col] = {}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforme un DataFrame en vecteur de caractéristiques pour le modèle."""
        if not self.is_fitted:
            raise RuntimeError("Le préprocesseur doit être ajusté (fit) avant d'être utilisé.")

        X = pd.DataFrame(index=df.index)

        # Variables numériques avec imputation
        for col in self.numerical_cols:
            if col in df.columns:
                X[col] = df[col].fillna(self.numerical_medians.get(col, 0.0)).astype(np.float32)
            else:
                X[col] = np.float32(self.numerical_medians.get(col, 0.0))

        # Variables catégorielles encodées
        for col in self.categorical_cols:
            mapping = self.category_mappings.get(col, {})
            if col in df.columns:
                series = df[col].astype(str)
                X[col] = series.map(mapping).fillna(0).astype(np.int32)
            else:
                X[col] = np.int32(0)

        # Réorganisation stricte selon l'ordre défini
        return X[self.feature_names]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajuste et transforme en une seule passe."""
        return self.fit(df).transform(df)

    def save(self, filepath: str = "data/models/preprocessor.joblib") -> None:
        """Enregistre le préprocesseur sérialisé."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[Preprocessor] Sauvegardé dans {filepath}")

    @classmethod
    def load(cls, filepath: str = "data/models/preprocessor.joblib") -> FraudDataPreprocessor:
        """Charge un préprocesseur sérialisé."""
        return joblib.load(filepath)


if __name__ == "__main__":
    print("[Preprocessor] Test rapide...")
    prep = FraudDataPreprocessor()
    print(f"Total features configurées : {len(prep.feature_names)}")
