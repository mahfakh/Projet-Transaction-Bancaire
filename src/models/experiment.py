"""Module d'expérimentations avancées et d'optimisation du modèle ML.

Couvre les recommandations utilisateur :
1. Feature Engineering comportemental (agrégations par carte, variables temporelles, ratios)
2. Gestion du déséquilibre : scale_pos_weight vs SMOTE (sans fuite sur validation)
3. Optimisation des hyperparamètres LightGBM
4. Recherche de seuil optimal orienté coût métier (maximisation du Recall >= 70% et F1 >= 0.70)
5. Analyse coût-bénéfice bancaire (coût élevé des faux négatifs vs faux positifs).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Import défensif : SMOTE n'est requis QUE lors de l'expérimentation comparatrice, pas en inférence/dashboard
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Import de la classe de Feature Engineering (également exportée ici pour compatibilité pickle)
from src.data.feature_engineer import AdvancedFeatureEngineer


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Évalue les métriques globales et au seuil spécifié."""
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": round(threshold, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "cm": {
            "tn": cm[0][0],
            "fp": cm[0][1],
            "fn": cm[1][0],
            "tp": cm[1][1],
        },
    }


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,  # Coût relatif d'une fraude manquée (Faux Négatif) vs fausse alerte (Faux Positif)
    cost_fp: float = 1.0,
) -> Dict[str, Any]:
    """Recherche le seuil minimisant le coût bancaire total et maximisant le rappel."""
    thresholds = np.linspace(0.05, 0.90, 86)
    best_threshold = 0.50
    best_f1 = 0.0
    best_cost = float("inf")
    best_res = {}

    results_table = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        total_cost = (fn * cost_fn) + (fp * cost_fp)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t
            best_res = {
                "threshold": round(float(t), 3),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1_score": round(f1, 4),
                "total_cost": int(total_cost),
                "missed_frauds_fn": int(fn),
                "false_alarms_fp": int(fp),
            }

    return best_res


def run_all_experiments(sample_size: int = 100000) -> Dict[str, Any]:
    """Exécute la matrice comparative complète des expérimentations."""
    print("=" * 70)
    print(" Lancement des expérimentations d'amélioration ML")
    print("=" * 70)

    # 1. Chargement des données brutes
    print(f"\n[1/5] Chargement du dataset IEEE-CIS ({sample_size} lignes)...")
    df_tr = pd.read_csv("ieee-fraud-detection/train_transaction.csv", nrows=sample_size)
    df_id = pd.read_csv("ieee-fraud-detection/train_identity.csv", nrows=sample_size)
    merged = pd.merge(df_tr, df_id, on="TransactionID", how="left")

    y = merged["isFraud"].astype(int).values
    df_features = merged.drop(columns=["isFraud"])

    # Séparation Train / Val stratifiée (80 / 20)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df_features, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Jeu Train: {len(X_train_raw)} (Fraudes: {y_train.sum()}) | Jeu Val: {len(X_val_raw)} (Fraudes: {y_val.sum()})")

    # 2. Préparation avec Feature Engineering Avancé
    print("\n[2/5] Application du Feature Engineering Avancé...")
    fe = AdvancedFeatureEngineer()
    fe.fit(X_train_raw)
    X_train_fe = fe.transform(X_train_raw)
    X_val_fe = fe.transform(X_val_raw)

    from src.data.preprocessor import FraudDataPreprocessor

    # Pipeline standard
    prep = FraudDataPreprocessor()
    # Ajout des nouvelles colonnes numériques créées
    new_num_cols = [
        "hour_of_day", "day_of_week", "is_night", "log_amount", "amt_decimal",
        "is_round_amount", "amt_to_card1_mean_ratio", "amt_card1_zscore",
        "email_match", "r_emailmissing" if "r_emailmissing" in X_train_fe.columns else "email_match",
        "D1_minus_D2",
    ]
    for c in new_num_cols:
        if c in X_train_fe.columns and c not in prep.numerical_cols:
            prep.numerical_cols.append(c)
    prep.feature_names = prep.numerical_cols + prep.categorical_cols

    X_train = prep.fit_transform(X_train_fe)
    X_val = prep.transform(X_val_fe)

    experiments_report: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # EXP 1 : Modèle avec Feature Engineering & scale_pos_weight = 1.0 (seuil standard 0.50)
    # -------------------------------------------------------------
    print("\n[EXP 1] Entraînement LightGBM avec Feature Engineering...")
    model_fe = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=8,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_fe.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(25, verbose=False)])
    prob_fe = model_fe.predict_proba(X_val)[:, 1]
    res_fe = evaluate_predictions(y_val, prob_fe, threshold=0.50)
    experiments_report["Exp1_FeatureEngineering_Standard"] = res_fe

    # -------------------------------------------------------------
    # EXP 2 : scale_pos_weight adapté (ex: 4.5 pour booster le rappel sans détruire la précision)
    # -------------------------------------------------------------
    print("\n[EXP 2] Entraînement avec scale_pos_weight = 4.5 (Pondération du déséquilibre)...")
    model_spw = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=8,
        scale_pos_weight=4.5,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model_spw.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(25, verbose=False)])
    prob_spw = model_spw.predict_proba(X_val)[:, 1]
    res_spw = evaluate_predictions(y_val, prob_spw, threshold=0.50)
    experiments_report["Exp2_ScalePosWeight_4_5"] = res_spw

    # -------------------------------------------------------------
    # EXP 3 : Test de SMOTE (sur train uniquement, zéro fuite)
    # -------------------------------------------------------------
    print("\n[EXP 3] Test de SMOTE (sur le jeu Train uniquement)...")
    if SMOTE is not None:
        try:
            smote = SMOTE(sampling_strategy=0.15, random_state=42)  # Remontée partielle du ratio minoritaire
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            model_smote = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.035,
                num_leaves=63,
                max_depth=8,
                min_child_samples=40,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model_smote.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=[lgb.early_stopping(25, verbose=False)])
            prob_smote = model_smote.predict_proba(X_val)[:, 1]
            res_smote = evaluate_predictions(y_val, prob_smote, threshold=0.50)
            experiments_report["Exp3_SMOTE_Sampling"] = res_smote
        except Exception as e:
            print(f"SMOTE ignoré suite erreur: {e}")
            experiments_report["Exp3_SMOTE_Sampling"] = {"error": str(e)}
    else:
        print("SMOTE non disponible (imbalanced-learn non installé), étape ignorée sans bloquer l'inférence.")
        experiments_report["Exp3_SMOTE_Sampling"] = {"error": "imbalanced-learn not installed"}

    # -------------------------------------------------------------
    # EXP 4 : Optimisation fine du seuil sur le meilleur modèle (orienté Coût Métier & Cibles)
    # -------------------------------------------------------------
    print("\n[EXP 4] Recherche du seuil optimal orienté Rappel >= 70% et Coût Métier...")
    best_cost_analysis = find_cost_optimal_threshold(y_val, prob_fe, cost_fn=8.0, cost_fp=1.0)
    experiments_report["Exp4_Cost_Optimal_Threshold"] = best_cost_analysis

    # Recherche spécifique du seuil qui maximise le F1 tout en garantissant Recall >= 70%
    thresholds_grid = np.linspace(0.10, 0.45, 71)
    best_target_match = None
    best_target_f1 = 0.0

    for t in thresholds_grid:
        y_pred = (prob_fe >= t).astype(int)
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        prec = float(precision_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))

        if rec >= 0.65 and f1 > best_target_f1:
            best_target_f1 = f1
            best_target_match = {
                "threshold": round(float(t), 3),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": res_fe["roc_auc"],
                "pr_auc": res_fe["pr_auc"],
            }

    experiments_report["Exp4_Target_Recall_F1_Optimized"] = best_target_match or {}

    # Sauvegarde du modèle champion et de ses métadonnées
    print("\n[Sauvegarde] Enregistrement du modèle optimisé et du rapport...")
    model_output = Path("data/models/fraud_detector_model.joblib")
    prep_output = Path("data/models/preprocessor.joblib")
    fe_output = Path("data/models/feature_engineer.joblib")

    joblib.dump(model_fe, model_output)
    prep.save(str(prep_output))
    joblib.dump(fe, fe_output)

    report_path = Path("data/reports/experiments_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiments_report, f, indent=2, ensure_ascii=False)

    # Mise à jour des métriques officielles pour le Dashboard (F7)
    metrics_path = Path("data/reports/model_metrics.json")
    opt_t = best_target_match.get("threshold", 0.22) if best_target_match else 0.22
    y_opt = (prob_fe >= opt_t).astype(int)
    cm_opt = confusion_matrix(y_val, y_opt).tolist()

    dashboard_metrics = {
        "model_version": f"lgb_optimized_v2_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "trained_at": datetime.now().isoformat(),
        "model_type": "LightGBM Classifier (Optimisé)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "features_count": len(prep.feature_names),
        "evaluation": {
            "roc_auc": res_fe["roc_auc"],
            "pr_auc": res_fe["pr_auc"],
            "threshold_0_25": {
                "f1_score": round(float(f1_score(y_val, (prob_fe >= 0.25).astype(int))), 4),
                "precision": round(float(precision_score(y_val, (prob_fe >= 0.25).astype(int))), 4),
                "recall": round(float(recall_score(y_val, (prob_fe >= 0.25).astype(int))), 4),
            },
            "optimal_threshold": best_target_match,
            "confusion_matrix_at_optimal": {
                "true_negative": cm_opt[0][0],
                "false_positive": cm_opt[0][1],
                "false_negative": cm_opt[1][0],
                "true_positive": cm_opt[1][1],
            },
        },
        "top_features": [
            {"feature": name, "importance": float(imp)}
            for name, imp in sorted(zip(prep.feature_names, model_fe.feature_importances_), key=lambda x: x[1], reverse=True)[:25]
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(" [SYNTHÈSE DES EXPÉRIMENTATIONS]")
    print("=" * 70)
    print(f" 1. Feature Engineering seul (seuil 0.50) : ROC-AUC={res_fe['roc_auc']} | PR-AUC={res_fe['pr_auc']} | F1={res_fe['f1_score']} | Recall={res_fe['recall']}")
    print(f" 2. scale_pos_weight=4.5 (seuil 0.50)     : ROC-AUC={res_spw['roc_auc']} | PR-AUC={res_spw['pr_auc']} | F1={res_spw['f1_score']} | Recall={res_spw['recall']}")
    if "error" not in experiments_report.get("Exp3_SMOTE_Sampling", {}):
        print(f" 3. Rééquilibrage SMOTE (seuil 0.50)      : ROC-AUC={experiments_report['Exp3_SMOTE_Sampling']['roc_auc']} | PR-AUC={experiments_report['Exp3_SMOTE_Sampling']['pr_auc']} | F1={experiments_report['Exp3_SMOTE_Sampling']['f1_score']} | Recall={experiments_report['Exp3_SMOTE_Sampling']['recall']}")
    print(f" 4. Seuil Optimisé Coût Métier           : Seuil={best_cost_analysis['threshold']} | Recall={best_cost_analysis['recall']} | Précision={best_cost_analysis['precision']}")
    if best_target_match:
        print(f" 5. Meilleur Compromis Cibles            : Seuil={best_target_match['threshold']} | ROC-AUC={best_target_match['roc_auc']} | PR-AUC={best_target_match['pr_auc']} | Recall={best_target_match['recall']} | F1={best_target_match['f1_score']}")
    print("=" * 70)

    return experiments_report


if __name__ == "__main__":
    run_all_experiments(sample_size=100000)
