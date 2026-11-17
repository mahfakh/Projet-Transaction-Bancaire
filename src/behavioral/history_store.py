"""Stockage de l'historique glissant par porteur de carte (F4).

Deux implémentations interchangeables :
- `InMemoryHistoryStore` : dictionnaire Python (mode local, tests). Perdu à l'arrêt du processus.
- `DynamoHistoryStore`   : table DynamoDB (mode live). Indispensable sur AWS Lambda, dont la
  mémoire est effacée entre deux exécutions : sans elle, chaque transaction serait vue comme
  la première de la carte et aucun signal comportemental ne se déclencherait.

Format d'un événement d'historique (types simples, sérialisables en JSON / DynamoDB) :
    {"transaction_id": str, "amount": float, "timestamp": str ISO 8601,
     "device": str, "location": str}

Concurrence : sur AWS, la clé de partition Kinesis est la clé du porteur ; toutes ses
transactions passent donc par le même shard et sont traitées dans l'ordre, une à la fois.
Il n'y a pas d'écritures concurrentes sur un même élément de la table.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.aws.adapter import from_dynamo, to_dynamo

# Un porteur inactif depuis 30 jours voit son historique supprimé automatiquement (TTL DynamoDB)
HISTORY_TTL_SECONDS = 30 * 24 * 3600


class InMemoryHistoryStore:
    """Historique en mémoire du processus (mode local)."""

    def __init__(self) -> None:
        self._data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def get(self, profile_key: str) -> List[Dict[str, Any]]:
        return list(self._data.get(profile_key, []))

    def put(self, profile_key: str, events: List[Dict[str, Any]]) -> None:
        self._data[profile_key] = list(events)


class DynamoHistoryStore:
    """Historique persistant dans DynamoDB : un élément par porteur (clé `profile_key`).

    Un seul élément contenant la liste des N derniers événements (N = 50 par défaut) :
    1 lecture + 1 écriture par transaction, très économique en mode on-demand.
    L'attribut `expires_at` (epoch) est utilisé par le TTL DynamoDB pour purger les profils inactifs.
    """

    def __init__(self, table_name: str, region: str, table: Any = None) -> None:
        if table is None:
            import boto3
            table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self.table = table

    def get(self, profile_key: str) -> List[Dict[str, Any]]:
        # Lecture fortement cohérente : on veut voir la dernière écriture, même toute récente
        resp = self.table.get_item(Key={"profile_key": profile_key}, ConsistentRead=True)
        item = resp.get("Item")
        return from_dynamo(item.get("events", [])) if item else []

    def put(self, profile_key: str, events: List[Dict[str, Any]]) -> None:
        self.table.put_item(Item=to_dynamo({
            "profile_key": profile_key,
            "events": events,
            "expires_at": int(time.time()) + HISTORY_TTL_SECONDS,
        }))


def build_history_store(mode: Optional[str] = None) -> Any:
    """Choisit l'implémentation selon le mode (local -> mémoire, live -> DynamoDB)."""
    from src.config import get_config

    cfg = get_config()
    if (mode or cfg.mode) == "live":
        return DynamoHistoryStore(table_name=cfg.card_history_table, region=cfg.aws_region)
    return InMemoryHistoryStore()
