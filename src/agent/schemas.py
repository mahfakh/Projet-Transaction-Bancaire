"""Contrats de données proposés (Section 4 du document de conception).

Définit les schémas stricts pour :
- Événement de transaction (F1)
- Résultat de détection ML & règles (F3)
- Investigation de l'agent LLM (F5)
- Alerte & Revue humaine (F6)
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    INVESTIGATE = "INVESTIGATE"
    HIGH = "HIGH"


class AlertStatus(str, Enum):
    PENDING = "PENDING"
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class TransactionEvent(BaseModel):
    """Événement de transaction brut ingéré (Section 4 & F1)."""
    transaction_id: str = Field(..., description="Identifiant unique de transaction")
    event_timestamp: str = Field(..., description="Horodatage ISO de l'événement")
    user_or_card_key: str = Field(..., description="Identifiant synthétique / pseudonymisé de carte ou compte")
    amount: float = Field(..., description="Montant de la transaction")
    currency: str = Field(default="USD", description="Devise (USD ou simulée)")
    card_type: Optional[str] = Field(default="visa", description="Réseau de carte (visa, mastercard, etc.)")
    device_info: Optional[str] = Field(default="Unknown", description="Information équipement / navigateur")
    ip_or_location: Optional[str] = Field(default=None, description="Localisation géographique ou adresse IP")
    source: str = Field(default="stream_generator", description="Origine (dataset ou générateur)")
    raw_features: Dict[str, Any] = Field(default_factory=dict, description="Caractéristiques IEEE-CIS associées")


class BehavioralSignals(BaseModel):
    """Signaux comportementaux calculés à partir de l'historique (F4)."""
    transactions_last_24h: int = Field(default=1, description="Nombre de transactions sur 24h")
    avg_amount_historical: float = Field(default=0.0, description="Montant moyen historique du compte")
    amount_ratio_to_avg: float = Field(default=1.0, description="Ratio du montant actuel / moyenne historique")
    is_amount_abnormal: bool = Field(default=False, description="Montant anormalement élevé (> 2.5 sigma)")
    device_changed: bool = Field(default=False, description="Changement soudain d'équipement")
    location_changed: bool = Field(default=False, description="Changement géographique inhabituel")
    synthetic_notes: List[str] = Field(default_factory=list, description="Mention explicite des signaux simulés")


class DetectionResult(BaseModel):
    """Résultat du scoring ML et des règles (Section 4 & F3)."""
    transaction_id: str
    model_version: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Probabilité de fraude ML")
    risk_level: RiskLevel
    behavioral_signals: BehavioralSignals
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    top_contributing_factors: List[str] = Field(default_factory=list)


class LLMInvestigation(BaseModel):
    """Sortie structurée de l'agent LLM d'investigation (Section 4 & F5)."""
    transaction_id: str
    investigation_summary: str = Field(..., description="Synthèse factuelle des signaux suspects")
    observed_risk_factors: List[str] = Field(default_factory=list, description="Facteurs de risque observés")
    missing_information: List[str] = Field(default_factory=list, description="Données absentes pour certifier")
    uncertainty: str = Field(default="MODERATE", description="Niveau d'incertitude (LOW, MODERATE, HIGH)")
    suggested_review: str = Field(..., description="Recommandation d'investigation pour l'analyste humain")
    disclaimer: str = Field(
        default="Avis généré automatiquement pour assistance humaine. Ne constitue pas un blocage automatique.",
        description="Mention obligatoire d'assistance non décisionnelle"
    )


class AlertReview(BaseModel):
    """Alerte pour supervision et revue humaine (Section 4 & F6)."""
    alert_id: str
    transaction_id: str
    risk_level: RiskLevel
    fraud_score: float
    status: AlertStatus = Field(default=AlertStatus.PENDING)
    analyst_decision: Optional[str] = None
    analyst_note: Optional[str] = None
    transaction_data: Dict[str, Any] = Field(default_factory=dict)
    detection_result: Dict[str, Any] = Field(default_factory=dict)
    llm_investigation: Optional[LLMInvestigation] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
