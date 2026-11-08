"""Module d'ingénierie des caractéristiques comportementales pour la détection de fraude.

Zéro dépendance externe lourde : uniquement numpy et pandas.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


class AdvancedFeatureEngineer:
    """Enrichissement des caractéristiques pour capturer les comportements suspects."""

    def __init__(self) -> None:
        self.card1_amt_means: Dict[Any, float] = {}
        self.card1_amt_stds: Dict[Any, float] = {}
        self.global_amt_mean: float = 135.0

    def fit(self, df: pd.DataFrame) -> AdvancedFeatureEngineer:
        """Calcule les agrégations historiques sur le train uniquement (zéro fuite de données)."""
        self.global_amt_mean = float(df["TransactionAmt"].mean())
        # Moyenne et écart-type par card1
        grouped = df.groupby("card1")["TransactionAmt"]
        self.card1_amt_means = grouped.mean().to_dict()
        self.card1_amt_stds = grouped.std().fillna(30.0).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'ingénierie des caractéristiques."""
        df_out = df.copy()

        # 1. Variables temporelles issues de TransactionDT (secondes écoulées)
        if "TransactionDT" in df_out.columns:
            dt = df_out["TransactionDT"].fillna(0)
            df_out["hour_of_day"] = ((dt // 3600) % 24).astype(np.float32)
            df_out["day_of_week"] = ((dt // (3600 * 24)) % 7).astype(np.float32)
            df_out["is_night"] = df_out["hour_of_day"].isin([0, 1, 2, 3, 4, 5]).astype(np.float32)
        else:
            df_out["hour_of_day"] = 12.0
            df_out["day_of_week"] = 3.0
            df_out["is_night"] = 0.0

        # 2. Décomposition du montant
        amt = df_out["TransactionAmt"].fillna(0.0)
        df_out["log_amount"] = np.log1p(amt).astype(np.float32)
        df_out["amt_decimal"] = (amt - np.floor(amt)).astype(np.float32)
        df_out["is_round_amount"] = (df_out["amt_decimal"] == 0.0).astype(np.float32)

        # 3. Ratios comportementaux par porteur de carte (card1)
        card1_series = df_out["card1"]
        mean_series = card1_series.map(self.card1_amt_means).fillna(self.global_amt_mean)
        std_series = card1_series.map(self.card1_amt_stds).fillna(30.0)

        df_out["amt_to_card1_mean_ratio"] = (amt / (mean_series + 1.0)).astype(np.float32)
        df_out["amt_card1_zscore"] = ((amt - mean_series) / (std_series + 1.0)).astype(np.float32)

        # 4. Cohérence des domaines de messagerie
        if "P_emaildomain" in df_out.columns and "R_emaildomain" in df_out.columns:
            p_email = df_out["P_emaildomain"].astype(str)
            r_email = df_out["R_emaildomain"].astype(str)
            df_out["email_match"] = ((p_email == r_email) & (p_email != "nan") & (p_email != "None")).astype(np.float32)
            df_out["r_email_missing"] = df_out["R_emaildomain"].isna().astype(np.float32)
        else:
            df_out["email_match"] = 0.0
            df_out["r_email_missing"] = 1.0

        # 5. Différences temporelles entre cartes (D-features)
        if "D1" in df_out.columns and "D2" in df_out.columns:
            df_out["D1_minus_D2"] = (df_out["D1"].fillna(0) - df_out["D2"].fillna(0)).astype(np.float32)

        return df_out
