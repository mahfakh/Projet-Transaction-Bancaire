"""Agent LLM d'investigation de fraude (F5).

Règles impératives :
- Déclenché uniquement lorsque le score ML ou les signaux requièrent une investigation
- Ne reçoit que les données strictement nécessaires et pseudonymisées (moindre privilège)
- Ne modifie JAMAIS le score ML
- Ne prend AUCUNE décision finale et n'effectue aucun blocage réel
- Produit une sortie rigoureusement structurée (LLMInvestigation) :
    * Synthèse factuelle
    * Facteurs de risque observés
    * Informations manquantes
    * Niveau d'incertitude
    * Recommandation pour l'analyste humain
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent.schemas import (
    BehavioralSignals,
    DetectionResult,
    LLMInvestigation,
    RiskLevel,
)


# Prompt système hautement contraint pour l'Agent LLM
SYSTEM_PROMPT = """Tu es un Agent IA spécialisé dans l'explication et l'investigation de signaux de fraude bancaire pour assister un analyste humain.
Règles strictes :
1. Tu ne prends AUCUNE décision de blocage ou de validation.
2. Tu ne dois JAMAIS inventer de données non fournies dans le contexte.
3. Tu ne dois JAMAIS modifier le score de risque calculé par le modèle ML.
4. Identifie clairement les éléments observés et cite explicitement les informations qui manquent pour confirmer ou infirmer la fraude.
5. Indique toujours ton niveau d'incertitude (LOW, MODERATE, HIGH).
6. Réponds impérativement au format JSON structuré selon les champs requis :
   - investigation_summary (string)
   - observed_risk_factors (list de strings)
   - missing_information (list de strings)
   - uncertainty (string : LOW, MODERATE ou HIGH)
   - suggested_review (string : recommandation d'action pour l'analyste)
"""


class LLMInvestigator:
    """Agent d'investigation contraint pour la revue des alertes (F5)."""

    def __init__(
        self,
        provider: str = "mock",  # "mock", "bedrock", "sagemaker"
        aws_region: str = "eu-west-3",
    ) -> None:
        self.provider = provider
        self.aws_region = aws_region

    def build_prompt_context(
        self,
        tx_data: Dict[str, Any],
        detection: DetectionResult,
    ) -> Dict[str, Any]:
        """Prépare un contexte minimal, pseudonymisé et nettoyé pour le LLM (F5 & F9)."""
        # Masquage et sélection stricte
        signals = detection.behavioral_signals
        context = {
            "transaction_id": detection.transaction_id,
            "fraud_score_ml": round(detection.fraud_score, 4),
            "risk_level": detection.risk_level.value,
            "amount": float(tx_data.get("TransactionAmt", tx_data.get("amount", 0.0))),
            "currency": str(tx_data.get("currency", "USD")),
            "card_network": str(tx_data.get("card4", "Inconnu")),
            "product_code": str(tx_data.get("ProductCD", "N/A")),
            "device_info": str(tx_data.get("DeviceInfo", tx_data.get("device_info", "Non renseigné"))),
            "email_domain": str(tx_data.get("P_emaildomain", "Non renseigné")),
            "behavioral_context": {
                "transactions_last_24h": signals.transactions_last_24h,
                "historical_mean_amount": signals.avg_amount_historical,
                "ratio_to_habitual_amount": signals.amount_ratio_to_avg,
                "is_amount_abnormal": signals.is_amount_abnormal,
                "device_changed": signals.device_changed,
                "location_changed": signals.location_changed,
            },
            "model_contributing_factors": detection.top_contributing_factors,
        }
        return context

    def investigate(
        self,
        tx_data: Dict[str, Any],
        detection: DetectionResult,
    ) -> LLMInvestigation:
        """Exécute l'investigation et retourne le rapport structuré."""
        context = self.build_prompt_context(tx_data, detection)

        if self.provider == "bedrock":
            return self._call_bedrock(context)
        else:
            return self._generate_structured_analysis(context)

    def _generate_structured_analysis(self, ctx: Dict[str, Any]) -> LLMInvestigation:
        """Générateur d'investigation déterministe et conforme aux contraintes (mode local sans frais)."""
        factors: List[str] = []
        missing: List[str] = []
        b_ctx = ctx["behavioral_context"]
        score = ctx["fraud_score_ml"]
        amt = ctx["amount"]

        # Analyse des signaux factuels observés
        if score >= 0.70:
            factors.append(f"Score ML critique ({score:.2f}) signalant une forte anomalie multifactorielle.")
        elif score >= 0.25:
            factors.append(f"Score ML modéré ({score:.2f}) au-dessus du seuil de surveillance.")

        if b_ctx["is_amount_abnormal"]:
            factors.append(
                f"Montant inhabituel ({amt:.2f} USD) représentant {b_ctx['ratio_to_habitual_amount']:.1f}x la moyenne historique ({b_ctx['historical_mean_amount']:.2f} USD)."
            )

        if b_ctx["transactions_last_24h"] > 3:
            factors.append(
                f"Fréquence élevée : {b_ctx['transactions_last_24h']} transactions enregistrées sur les dernières 24h (vélocité anormale)."
            )

        if b_ctx["device_changed"]:
            factors.append(f"Changement d'équipement de connexion détecté ({ctx['device_info']}).")

        email = ctx["email_domain"]
        if email not in ["Non renseigné", "gmail.com", "yahoo.com", "hotmail.com"]:
            factors.append(f"Domaine de messagerie atypique ({email}).")

        # Identification objective des informations manquantes
        missing.append("Confirmation directe du porteur de carte (contact client / 3-D Secure non renseigné).")
        if ctx["device_info"] in ["Non renseigné", "Unknown", "nan"]:
            missing.append("Détails d'empreinte numérique du terminal (Device Fingerprint absent).")
        missing.append("Historique d'adresses de livraison antérieures pour comparaison avec addr1.")

        # Détermination du niveau d'incertitude
        if score >= 0.80 and b_ctx["is_amount_abnormal"]:
            uncertainty = "LOW"
            summary = (
                f"La transaction {ctx['transaction_id']} présente une convergence forte de signaux d'alerte : "
                f"un score ML de {score:.2f} conjugué à un saut de montant de {b_ctx['ratio_to_habitual_amount']:.1f}x "
                f"par rapport aux habitudes du compte et une vélocité inhabituelle."
            )
            suggested = "Contacter immédiatement le porteur par SMS/App pour authentification avant confirmation commerciale."
        elif score >= 0.35 or b_ctx["is_amount_abnormal"]:
            uncertainty = "MODERATE"
            summary = (
                f"La transaction {ctx['transaction_id']} présente des écarts notables par rapport à l'historique "
                f"(montant de {amt:.2f} USD vs moyenne de {b_ctx['historical_mean_amount']:.2f} USD), "
                f"avec un score de risque ML intermédiaire de {score:.2f}."
            )
            suggested = "Vérifier la concordance de l'adresse de facturation et solliciter une validation renforcée si nécessaire."
        else:
            uncertainty = "HIGH"
            summary = (
                f"La transaction {ctx['transaction_id']} présente un score de risque ML faible ({score:.2f}) "
                f"sans rupture comportementale majeure constatée."
            )
            suggested = "Revue de routine sans blocage requis."

        if not factors:
            factors.append("Aucun facteur critique détecté sur les caractéristiques observées.")

        return LLMInvestigation(
            transaction_id=ctx["transaction_id"],
            investigation_summary=summary,
            observed_risk_factors=factors,
            missing_information=missing,
            uncertainty=uncertainty,
            suggested_review=suggested,
        )

    def _call_bedrock(self, ctx: Dict[str, Any]) -> LLMInvestigation:
        """Appel AWS Bedrock (Claude 3 / Llama 3) lorsque les identifiants AWS sont actifs."""
        try:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=self.aws_region)
            prompt_content = f"{SYSTEM_PROMPT}\n\nDonnées de transaction pour analyse :\n{json.dumps(ctx, ensure_ascii=False)}"
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 700,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt_content}],
            })
            resp = client.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=body,
            )
            resp_body = json.loads(resp["body"].read())
            raw_text = resp_body["content"][0]["text"]
            parsed = json.loads(raw_text)
            return LLMInvestigation(
                transaction_id=ctx["transaction_id"],
                investigation_summary=parsed.get("investigation_summary", ""),
                observed_risk_factors=parsed.get("observed_risk_factors", []),
                missing_information=parsed.get("missing_information", []),
                uncertainty=parsed.get("uncertainty", "MODERATE"),
                suggested_review=parsed.get("suggested_review", ""),
            )
        except Exception as e:
            print(f"[LLM Agent] Fallback vers analyse locale (Bedrock indisponible: {e})")
            return self._generate_structured_analysis(ctx)


if __name__ == "__main__":
    from src.agent.schemas import BehavioralSignals, RiskLevel

    agent = LLMInvestigator(provider="mock")
    tx_dummy = {
        "TransactionID": "TX_TEST_999",
        "TransactionAmt": 1850.0,
        "card4": "mastercard",
        "DeviceInfo": "SM-G960F",
        "P_emaildomain": "mail2world.com",
    }
    detection_dummy = DetectionResult(
        transaction_id="TX_TEST_999",
        model_version="lgb_v20261105",
        fraud_score=0.86,
        risk_level=RiskLevel.HIGH,
        behavioral_signals=BehavioralSignals(
            transactions_last_24h=5,
            avg_amount_historical=120.0,
            amount_ratio_to_avg=15.4,
            is_amount_abnormal=True,
            device_changed=True,
        ),
        top_contributing_factors=["Montant anormal", "Changement device"],
    )

    inv = agent.investigate(tx_dummy, detection_dummy)
    print("\n[LLM Investigation Output]")
    print(inv.model_dump_json(indent=2))
