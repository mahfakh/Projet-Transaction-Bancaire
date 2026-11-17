"""Validation des événements de transaction entrants (F1).

Module volontairement sans dépendance lourde : il est utilisé à la fois par le générateur
(côté producteur) et par la Lambda de scoring (côté consommateur Kinesis), qui revalide
chaque enregistrement avant traitement et envoie les événements invalides en Dead-Letter Queue.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

REQUIRED_FIELDS = ("TransactionAmt",)
MAX_AMOUNT = 1_000_000.0  # garde-fou contre les montants aberrants


def validate_event(raw_data: Any) -> Tuple[bool, Optional[str]]:
    """Retourne (True, None) si l'événement est conforme, sinon (False, raison du rejet)."""
    if not isinstance(raw_data, dict):
        return False, "L'événement n'est pas un objet JSON"

    for field in REQUIRED_FIELDS:
        if field not in raw_data or raw_data[field] is None:
            return False, f"Champ obligatoire manquant : {field}"

    try:
        amt = float(raw_data["TransactionAmt"])
    except (ValueError, TypeError):
        return False, "Format de montant non numérique"
    if math.isnan(amt) or amt <= 0:
        return False, f"Montant invalide ou négatif : {amt}"
    if amt > MAX_AMOUNT:
        return False, f"Montant hors limites : {amt}"

    return True, None
