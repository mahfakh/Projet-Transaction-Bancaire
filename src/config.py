"""Configuration centralisée de l'application.

Ordre de priorité (du plus fort au plus faible) :
1. Variables d'environnement (injectées par la Lambda / le template AWS, ou un fichier .env)
2. config/settings.yaml
3. Valeurs par défaut ci-dessous

Ainsi, le même code tourne en local (APP_MODE=local) et sur AWS (APP_MODE=live)
sans aucune modification : seuls les noms de ressources changent via l'environnement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"


@dataclass(frozen=True)
class AppConfig:
    mode: str  # "local" (stockage fichier, sans frais) ou "live" (services AWS réels)
    aws_region: str
    alerts_table: str
    transactions_table: str
    card_history_table: str
    s3_bucket: str
    kinesis_stream: str
    metrics_namespace: str
    llm_provider: str  # "mock" (analyse déterministe) ou "bedrock"
    bedrock_model_id: str
    llm_temperature: float
    llm_max_tokens: int
    storage_dir: Path

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _env(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@lru_cache(maxsize=None)
def get_config(settings_path: str | None = None) -> AppConfig:
    """Construit la configuration (mise en cache : lue une seule fois par processus)."""
    raw = _load_yaml(Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH)
    aws = raw.get("aws", {})
    agent = raw.get("agent", {})

    mode = str(_env("APP_MODE", aws.get("mode", "local"))).lower()
    if mode not in ("local", "live"):
        raise ValueError(f"APP_MODE invalide : {mode!r} (attendu : 'local' ou 'live')")

    return AppConfig(
        mode=mode,
        aws_region=_env("AWS_REGION", _env("AWS_DEFAULT_REGION", aws.get("region", "us-east-1"))),
        alerts_table=_env("ALERTS_TABLE", aws.get("dynamodb_alerts_table", "FraudDetectionAlerts")),
        transactions_table=_env("TRANSACTIONS_TABLE", aws.get("dynamodb_transactions_table", "FraudDetectionTransactions")),
        card_history_table=_env("CARD_HISTORY_TABLE", aws.get("dynamodb_card_history_table", "FraudDetectionCardHistory")),
        s3_bucket=_env("S3_BUCKET", aws.get("s3_bucket", "")),
        kinesis_stream=_env("KINESIS_STREAM", aws.get("kinesis_stream_name", "banking-transactions-stream")),
        metrics_namespace=_env("METRICS_NAMESPACE", aws.get("metrics_namespace", "BankingFraudDetection")),
        llm_provider=str(_env("LLM_PROVIDER", agent.get("provider", "mock"))).lower(),
        bedrock_model_id=_env("BEDROCK_MODEL_ID", agent.get("model_id", "")),
        llm_temperature=float(_env("LLM_TEMPERATURE", agent.get("temperature", 0.1))),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", agent.get("max_tokens", 700))),
        storage_dir=Path(_env("LOCAL_STORAGE_DIR", str(BASE_DIR / "data" / "storage"))),
    )
