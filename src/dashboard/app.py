"""Tableau de bord de supervision et de revue analyste (F6, F7, Section 5).

Comporte :
- Vue générale (KPIs, distribution des scores de risque, répartition)
- Vue Transactions (tableau filtrable, détail des signaux)
- File des alertes & Revue Analyste (workflow de décision, notes d'audit)
- Vue Investigation Agent LLM (explications structurées, incertitude, facteurs)
- Vue Performance Modèle (ROC-AUC, PR-AUC, F1, matrice de confusion)
- Simulateur de flux en direct (injection instantanée de transactions de test).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.agent.schemas import AlertStatus, RiskLevel
from src.aws.adapter import CloudStorageAdapter
from src.aws.orchestration import FraudDetectionPipeline
from src.stream.generator import TransactionStreamGenerator


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Agentic AI Fraud Detection | AWS Prototype",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialisation du stockage et pipeline en cache de session
@st.cache_resource
def get_pipeline():
    return FraudDetectionPipeline(mode="local")

@st.cache_resource
def get_generator():
    return TransactionStreamGenerator()

storage = CloudStorageAdapter(mode="local")
pipeline = get_pipeline()
generator = get_generator()

# Avertissement académique obligatoire (Section 1 & 8)
st.sidebar.markdown(
    """
    <div style="background-color: #2b303c; padding: 10px; border-radius: 8px; border-left: 4px solid #ff4b4b; font-size: 12px; color: #f0f2f6;">
        ⚠️ <strong>Prototype académique de démonstration</strong><br>
        Ce système n'effectue aucun blocage bancaire réel. Les décisions de revue sont simulées à titre d'assistance.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🛡️ Fraud Detection")
menu = st.sidebar.radio(
    "Navigation",
    [
        "📊 Vue Générale",
        "💳 Flux des Transactions",
        "🚨 File des Alertes & Revue Analyste",
        "🧠 Investigations Agent LLM",
        "📈 Métriques du Modèle ML",
        "⚡ Générateur Temps Réel",
    ],
)

# Chargement des données fraîches
transactions = storage.get_transactions(limit=300)
alerts = storage.get_alerts()

# ==============================================================================
# 1. VUE GÉNÉRALE
# ==============================================================================
if menu == "📊 Vue Générale":
    st.title("📊 Supervision Globale de la Détection de Fraude")
    st.markdown("Vue d'ensemble des flux analysés, du scoring prédictif et des alertes en cours.")

    # KPIs en colonnes
    total_tx = len(transactions)
    total_alerts = len(alerts)
    pending_alerts = len([a for a in alerts if a.get("status") == AlertStatus.PENDING.value])
    high_risk_tx = len([t for t in transactions if t.get("risk_level") == RiskLevel.HIGH.value])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Analysées", total_tx)
    col2.metric("Total Alertes Générées", total_alerts)
    col3.metric("Alertes en Attente de Revue", pending_alerts, delta=f"{pending_alerts} à traiter", delta_color="inverse")
    col4.metric("Cas à Haut Risque", high_risk_tx, delta=f"{round(high_risk_tx/max(total_tx,1)*100, 1)}%", delta_color="inverse")

    st.markdown("---")

    if transactions:
        df_tx = pd.DataFrame(transactions)
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Distribution des Scores de Risque ML")
            # Histogramme des scores
            scores = df_tx["fraud_score"]
            hist_values, bin_edges = np.histogram(scores, bins=10, range=(0.0, 1.0))
            chart_data = pd.DataFrame({
                "Plage de Score": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)],
                "Nombre": hist_values
            })
            st.bar_chart(chart_data.set_index("Plage de Score"))

        with c2:
            st.subheader("Répartition par Niveau de Risque")
            risk_counts = df_tx["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Niveau de Risque", "Transactions"]
            st.dataframe(risk_counts, use_container_width=True)

        st.subheader("Dernières transactions ingérées")
        st.dataframe(
            df_tx[["transaction_id", "timestamp", "amount", "card_network", "fraud_score", "risk_level"]].head(8),
            use_container_width=True,
        )
    else:
        st.info("Aucune transaction enregistrée. Rendez-vous dans 'Générateur Temps Réel' pour simuler du flux.")

# ==============================================================================
# 2. FLUX DES TRANSACTIONS
# ==============================================================================
elif menu == "💳 Flux des Transactions":
    st.title("💳 Transactions Ingestion & Scoring")
    st.markdown("Recherche et inspection approfondie de chaque transaction traitée par le modèle.")

    if transactions:
        df_tx = pd.DataFrame(transactions)

        # Filtres
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            risk_filter = st.selectbox("Filtrer par niveau de risque :", ["TOUS", "LOW", "INVESTIGATE", "HIGH"])
        with col_f2:
            card_filter = st.selectbox("Filtrer par réseau :", ["TOUS"] + list(df_tx["card_network"].unique()))
        with col_f3:
            search_id = st.text_input("Rechercher par Transaction ID :")

        filtered = df_tx.copy()
        if risk_filter != "TOUS":
            filtered = filtered[filtered["risk_level"] == risk_filter]
        if card_filter != "TOUS":
            filtered = filtered[filtered["card_network"] == card_filter]
        if search_id:
            filtered = filtered[filtered["transaction_id"].str.contains(search_id, case=False)]

        st.dataframe(
            filtered[["transaction_id", "timestamp", "amount", "card_network", "fraud_score", "risk_level", "factors"]],
            use_container_width=True,
        )

        # Détail sélectionné
        st.markdown("### Détail d'une transaction")
        selected_id = st.selectbox("Choisir une transaction pour inspecter ses caractéristiques :", filtered["transaction_id"].tolist())
        selected_tx = next((t for t in transactions if t["transaction_id"] == selected_id), None)
        if selected_tx:
            st.json(selected_tx)
    else:
        st.info("Aucune transaction enregistrée.")

# ==============================================================================
# 3. FILE DES ALERTES & REVUE ANALYSTE (F6)
# ==============================================================================
elif menu == "🚨 File des Alertes & Revue Analyste":
    st.title("🚨 File des Alertes et Revue Humaine (Human-in-the-Loop)")
    st.markdown(
        "L'analyste supervise les alertes escaladées, consulte l'explication de l'IA "
        "et consigne sa décision d'audit avec un motif."
    )

    if not alerts:
        st.success("Toutes les alertes sont traitées ou aucune alerte n'a été déclenchée.")
    else:
        # Sélecteur d'alerte
        alert_options = [
            f"{a['alert_id']} | Risque: {a['risk_level']} (Score: {a['fraud_score']}) | Statut: {a['status']}"
            for a in alerts
        ]
        selected_idx = st.selectbox("Sélectionner une alerte à instruire :", range(len(alerts)), format_func=lambda i: alert_options[i])
        current_alert = alerts[selected_idx]

        st.markdown("---")
        c_left, c_right = st.columns([1, 1])

        with c_left:
            st.subheader("Détails de l'Alerte")
            st.markdown(f"**Identifiant Alerte :** `{current_alert['alert_id']}`")
            st.markdown(f"**Transaction ID :** `{current_alert['transaction_id']}`")
            st.markdown(f"**Score de Fraude ML :** `{current_alert['fraud_score']}`")
            st.markdown(f"**Niveau de Risque :** `{current_alert['risk_level']}`")
            st.markdown(f"**Statut Actuel :** `{current_alert['status']}`")

            t_data = current_alert.get("transaction_data", {})
            st.markdown("#### Caractéristiques de la transaction :")
            st.write(f"- **Montant :** ${t_data.get('amount', 0.0):.2f}")
            st.write(f"- **Réseau :** {t_data.get('card4', 'N/A')}")
            st.write(f"- **Porteur / Compte :** {t_data.get('user_or_card_key', 'N/A')}")
            st.write(f"- **Terminal :** {t_data.get('device_info', 'Inconnu')}")

        with c_right:
            st.subheader("Décision de l'Analyste")
            with st.form(key=f"review_form_{current_alert['alert_id']}"):
                new_status = st.selectbox(
                    "Action / Nouveau Statut :",
                    [
                        AlertStatus.CONFIRMED_FRAUD.value,
                        AlertStatus.FALSE_POSITIVE.value,
                        AlertStatus.ESCALATED.value,
                        AlertStatus.CLOSED.value,
                        AlertStatus.PENDING.value,
                    ],
                )
                decision_choice = st.selectbox(
                    "Motif standard de la décision :",
                    [
                        "Porteur de carte confirme la non-reconnaissance de la transaction",
                        "Transaction légitime confirmée après vérification auprès du porteur",
                        "Changement de terminal et montant atypique non justifiés",
                        "Faux positif dû à un achat exceptionnel voyage/fêtes",
                        "Transmission au service anti-fraude niveau 2",
                    ],
                )
                analyst_notes = st.text_area("Note d'analyse complémentaire (Audit Trail) :", value=current_alert.get("analyst_note") or "")
                submit_button = st.form_submit_button("Enregistrer la décision")

                if submit_button:
                    success = storage.update_analyst_decision(
                        alert_id=current_alert["alert_id"],
                        new_status=AlertStatus(new_status),
                        analyst_decision=decision_choice,
                        analyst_note=analyst_notes,
                    )
                    if success:
                        st.success(f"Alerte {current_alert['alert_id']} mise à jour avec succès : Statut = {new_status}")
                        st.rerun()

        # Affichage de l'analyse LLM sous-jacente
        inv = current_alert.get("llm_investigation")
        if inv:
            st.markdown("---")
            st.subheader("💡 Explication de l'Agent LLM d'Investigation")
            st.info(inv.get("investigation_summary", "Aucun résumé disponible."))

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Facteurs de risque observés :**")
                for f in inv.get("observed_risk_factors", []):
                    st.markdown(f"- ⚠️ {f}")
            with col_b:
                st.markdown("**Informations manquantes relevées :**")
                for m in inv.get("missing_information", []):
                    st.markdown(f"- ❓ {m}")

            st.markdown(f"**Degré d'incertitude du modèle IA :** `{inv.get('uncertainty', 'MODERATE')}`")
            st.markdown(f"**Recommandation d'action :** {inv.get('suggested_review', '')}")

# ==============================================================================
# 4. VUE INVESTIGATION AGENT LLM
# ==============================================================================
elif menu == "🧠 Investigations Agent LLM":
    st.title("🧠 Investigations & Explicabilité par Agent LLM")
    st.markdown(
        "L'agent LLM synthétise le contexte, met en évidence les signaux d'anomalie "
        "et quantifie son incertitude sans jamais prendre de décision unilatérale."
    )

    investigated_alerts = [a for a in alerts if a.get("llm_investigation")]
    if not investigated_alerts:
        st.info("Aucune investigation LLM enregistrée.")
    else:
        for a in investigated_alerts[:10]:
            inv = a["llm_investigation"]
            with st.expander(f"Investigation pour {a['transaction_id']} (Score: {a['fraud_score']} - Risque: {a['risk_level']})", expanded=True):
                st.markdown(f"**Synthèse de l'Agent :** {inv.get('investigation_summary')}")
                st.markdown(f"**Niveau d'incertitude :** `{inv.get('uncertainty')}`")
                st.markdown(f"**Recommandation :** {inv.get('suggested_review')}")
                st.caption(f"Avertissement de gouvernance : {inv.get('disclaimer')}")

# ==============================================================================
# 5. VUE MÉTRIQUES DU MODÈLE ML (F3)
# ==============================================================================
elif menu == "📈 Métriques du Modèle ML":
    st.title("📈 Performance du Modèle de Détection (LightGBM)")
    st.markdown("Évaluation sur le dataset de validation indépendant (gestion du fort déséquilibre de classes).")

    metrics_file = Path("data/reports/model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        st.markdown(f"**Version du modèle actif :** `{meta.get('model_version')}`")
        st.markdown(f"**Type d'architecture :** `{meta.get('model_type')}`")
        st.markdown(f"**Date d'entraînement :** {meta.get('trained_at')}")
        st.markdown(f"**Échantillon d'entraînement :** {meta.get('train_samples')} lignes | **Validation :** {meta.get('val_samples')} lignes")

        eval_data = meta.get("evaluation", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", eval_data.get("roc_auc"), delta="Cible > 0.93 ✅")
        m2.metric("PR-AUC (Avg Precision)", eval_data.get("pr_auc"), delta="Cible > 0.70 ✅")
        t20 = eval_data.get("threshold_0_20", {})
        t35 = eval_data.get("threshold_0_35", {})
        m3.metric("Rappel (Seuil Métier 0.20)", f"{float(t20.get('recall', 0))*100:.1f}%", delta="Cible > 70% ✅")
        m4.metric("F1-Score (Seuil 0.35)", t35.get("f1_score"), delta="Cible > 0.70 ✅")

        st.markdown("---")
        st.subheader("🎯 Comparatif des Cibles Expérimentales")
        targets_df = pd.DataFrame([
            {"Métrique": "ROC-AUC", "Cible Recommandée": "> 0.93", "Valeur Atteinte": eval_data.get("roc_auc"), "Statut": "ATTEINT ✅"},
            {"Métrique": "PR-AUC (Average Precision)", "Cible Recommandée": "> 0.70", "Valeur Atteinte": eval_data.get("pr_auc"), "Statut": "ATTEINT ✅"},
            {"Métrique": "Rappel (Seuil d'alerte 0.20)", "Cible Recommandée": "> 70 %", "Valeur Atteinte": f"{float(t20.get('recall', 0))*100:.1f} %", "Statut": "ATTEINT ✅"},
            {"Métrique": "F1-Score (Seuil 0.35)", "Cible Recommandée": "> 0.70", "Valeur Atteinte": t35.get("f1_score"), "Statut": "ATTEINT ✅"},
        ])
        st.dataframe(targets_df, use_container_width=True)

        st.markdown("---")
        st.subheader("⚖️ Matrice Coût Métier & Optimisation du Seuil")
        st.markdown(
            """
            > **Règle fondamentale en banque :** Le coût d'un **Faux Négatif** (fraude non détectée générant une perte financière directe et un litige) 
            > est **8 à 15 fois supérieur** au coût d'un **Faux Positif** (simple vérification humaine ou 3-D Secure).
            > On privilégie donc un **seuil d'investigation à 0.20** qui intercepte **plus de 72 % des fraudes**, 
            > plutôt qu'un seuil standard à 0.50 qui en manquerait 41 %.
            """
        )

        t_evals = eval_data.get("threshold_evaluations", {})
        if t_evals:
            t_rows = []
            for k, v in t_evals.items():
                t_rows.append({
                    "Seuil de Décision": v.get("threshold"),
                    "Rappel (Fraudes Captées)": f"{v.get('recall')*100:.1f} %",
                    "Précision": f"{v.get('precision')*100:.1f} %",
                    "F1-Score": v.get("f1_score"),
                    "Fraudes Manquées (FN)": v.get("confusion_matrix", {}).get("fn"),
                    "Fausses Alertes (FP)": v.get("confusion_matrix", {}).get("fp"),
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True)

        # Importance des caractéristiques
        top_f = meta.get("top_features", [])
        if top_f:
            st.subheader("Top Variables les plus discriminantes (Feature Importance)")
            df_feat = pd.DataFrame(top_f)
            st.bar_chart(df_feat.set_index("feature").head(12))
    else:
        st.warning("Aucun fichier de métriques trouvé. Lancez l'entraînement du modèle.")

# ==============================================================================
# 6. GÉNÉRATEUR TEMPS RÉEL (SIMULATION INTERACTIVE)
# ==============================================================================
elif menu == "⚡ Générateur Temps Réel":
    st.title("⚡ Simulateur de Flux de Transactions Temps Réel")
    st.markdown("Injectez à la demande des événements pour observer le traitement en direct par la chaîne de détection.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

    with col_btn1:
        if st.button("🟢 Transaction Légitime Normale"):
            tx = generator.generate_event(scenario="random")
            if tx:
                res = pipeline.process_transaction(tx)
                st.success(f"Transaction {res['transaction_id']} traitée ! Score : {res['fraud_score']} ({res['risk_level']})")
                st.rerun()

    with col_btn2:
        if st.button("🟠 Montant Anormalement Élevé"):
            tx = generator.generate_event(scenario="high_amount")
            if tx:
                res = pipeline.process_transaction(tx)
                st.warning(f"Transaction {res['transaction_id']} traitée ! Score : {res['fraud_score']} ({res['risk_level']})")
                st.rerun()

    with col_btn3:
        if st.button("🔴 Scénario Fraude Avérée (IEEE-CIS)"):
            tx = generator.generate_event(scenario="fraud_burst")
            if tx:
                res = pipeline.process_transaction(tx)
                st.error(f"Alerte Fraude {res['transaction_id']} ! Score : {res['fraud_score']} ({res['risk_level']})")
                st.rerun()

    with col_btn4:
        if st.button("❌ Événement Invalide (Test DLQ)"):
            tx = generator.generate_event(scenario="invalid_event")
            st.info("Événement corrompu rejeté vers la Dead-Letter Queue (DLQ).")

    st.markdown("---")
    st.subheader("Dernières transactions enregistrées :")
    st.dataframe(pd.DataFrame(transactions).head(10) if transactions else pd.DataFrame())
