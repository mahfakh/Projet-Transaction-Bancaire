"""Script de Déploiement Automatisé Complet sur AWS (Boto3).

Automatise :
1. La vérification des identifiants AWS (IAM)
2. Le packaging des 5 fonctions Lambda en archives ZIP
3. La création / mise à jour de la pile CloudFormation (Kinesis, DynamoDB, S3, SNS, API Gateway, Lambdas)
4. Le téléversement des modèles ML LightGBM (.joblib) dans le bucket Amazon S3
5. La mise à jour du code des 5 fonctions Lambda
6. La mise à jour automatique de config/settings.yaml avec les endpoints live.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Chargement optionnel de .env
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import boto3
import yaml

REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-3")
STACK_NAME = os.environ.get("CFN_STACK_NAME", "fraud-detection-stack-prototype")


def check_aws_credentials() -> bool:
    """Vérifie la validité des clés AWS."""
    print("=" * 70)
    print(" [1/5] Vérification des identifiants AWS...")
    print("=" * 70)
    try:
        sts = boto3.client("sts", region_name=REGION)
        caller = sts.get_caller_identity()
        print(f" Connecté avec succès au compte AWS : {caller['Account']}")
        print(f" Identité IAM : {caller['Arn']}")
        print(f" Région active : {REGION}")
        return True
    except Exception as e:
        print("\n [ERREUR CRITIQUE] Impossible de se connecter à AWS.")
        print(f" Message : {e}")
        print("\n Instructions pour configurer vos clés AWS :")
        print(" 1. Ouvrez le fichier .env (ou créez-le à partir de .env.example)")
        print(" 2. Renseignez vos accès :")
        print("    AWS_ACCESS_KEY_ID=AKIA...")
        print("    AWS_SECRET_ACCESS_KEY=wJalr...")
        print("    AWS_DEFAULT_REGION=eu-west-3")
        print(" 3. Relancez ce script : python scripts/deploy_aws.py\n")
        return False


def create_lambda_zip(lambda_dir: Path) -> bytes:
    """Crée une archive zip en mémoire contenant le code de la fonction Lambda."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in lambda_dir.rglob("*.py"):
            zf.write(file, arcname=file.name)
    buf.seek(0)
    return buf.read()


def deploy_cloudformation_stack() -> Dict[str, str]:
    """Déploie le template CloudFormation via boto3."""
    print("\n" + "=" * 70)
    print(f" [2/5] Déploiement de la pile CloudFormation ({STACK_NAME})...")
    print("=" * 70)

    cfn = boto3.client("cloudformation", region_name=REGION)
    template_path = BASE_DIR / "infrastructure" / "cloudformation_template.yaml"

    with open(template_path, "r", encoding="utf-8") as f:
        template_body = f.read()

    stack_exists = False
    try:
        cfn.describe_stacks(StackName=STACK_NAME)
        stack_exists = True
    except Exception:
        pass

    if stack_exists:
        print(f" Mise à jour de la pile existante {STACK_NAME}...")
        try:
            cfn.update_stack(
                StackName=STACK_NAME,
                TemplateBody=template_body,
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
            )
        except Exception as e:
            if "No updates are to be performed" in str(e):
                print(" Aucun changement détecté dans le template.")
            else:
                raise e
    else:
        print(f" Création de la nouvelle pile {STACK_NAME}...")
        cfn.create_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"],
        )

    print(" Attente de la fin du déploiement (peut prendre 2 à 3 minutes)...")
    waiter = cfn.get_waiter("stack_update_complete" if stack_exists else "stack_create_complete")
    waiter.wait(StackName=STACK_NAME)

    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = {}
    for item in resp["Stacks"][0].get("Outputs", []):
        outputs[item["OutputKey"]] = item["OutputValue"]
        print(f" -> {item['OutputKey']} : {item['OutputValue']}")

    return outputs


def upload_models_to_s3(bucket_name: str) -> None:
    """Téléverse le modèle LightGBM et le préprocesseur sur Amazon S3."""
    print("\n" + "=" * 70)
    print(f" [3/5] Téléversement des artefacts ML vers S3 ({bucket_name})...")
    print("=" * 70)

    s3 = boto3.client("s3", region_name=REGION)
    models_dir = BASE_DIR / "data" / "models"

    for file_name in ["fraud_detector_model.joblib", "preprocessor.joblib", "feature_engineer.joblib"]:
        local_path = models_dir / file_name
        if local_path.exists():
            s3_key = f"models/{file_name}"
            print(f" Téléversement de {file_name} ({local_path.stat().st_size / 1024:.1f} KB)...")
            s3.upload_file(str(local_path), bucket_name, s3_key)
            print(f" ✅ s3://{bucket_name}/{s3_key}")
        else:
            print(f" ⚠️ Fichier {file_name} absent de {models_dir}. Lancez d'abord python src/models/train.py.")


def update_lambda_codes(env_name: str = "prototype") -> None:
    """Met à jour le code des 5 fonctions Lambda déployées."""
    print("\n" + "=" * 70)
    print(" [4/5] Mise à jour du code des fonctions AWS Lambda...")
    print("=" * 70)

    lambda_client = boto3.client("lambda", region_name=REGION)
    lambdas_base = BASE_DIR / "infrastructure" / "lambdas"

    mapping = {
        f"FraudDetection-Validator-{env_name}": lambdas_base / "validator",
        f"FraudDetection-Behavioral-{env_name}": lambdas_base / "behavioral",
        f"FraudDetection-Scorer-{env_name}": lambdas_base / "scorer",
        f"FraudDetection-Investigator-{env_name}": lambdas_base / "investigator",
        f"FraudDetection-AnalystApi-{env_name}": lambdas_base / "analyst_api",
    }

    for fn_name, pth in mapping.items():
        if pth.exists():
            print(f" Packaging & déploiement de {fn_name}...")
            zip_bytes = create_lambda_zip(pth)
            lambda_client.update_function_code(
                FunctionName=fn_name,
                ZipFile=zip_bytes,
            )
            print(f" ✅ {fn_name} à jour.")


def update_project_settings(outputs: Dict[str, str]) -> None:
    """Met à jour le fichier config/settings.yaml avec les ressources AWS réelles."""
    print("\n" + "=" * 70)
    print(" [5/5] Mise à jour de config/settings.yaml...")
    print("=" * 70)

    cfg_path = BASE_DIR / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["aws"]["mode"] = "live"
    cfg["aws"]["region"] = REGION
    if "ArtifactsS3BucketName" in outputs:
        cfg["aws"]["s3_bucket"] = outputs["ArtifactsS3BucketName"]
    if "AlertsDynamoTableName" in outputs:
        cfg["aws"]["dynamodb_alerts_table"] = outputs["AlertsDynamoTableName"]
    if "TransactionsDynamoTableName" in outputs:
        cfg["aws"]["dynamodb_transactions_table"] = outputs["TransactionsDynamoTableName"]
    if "KinesisStreamName" in outputs:
        cfg["aws"]["kinesis_stream_name"] = outputs["KinesisStreamName"]
    if "ApiEndpointUrl" in outputs:
        cfg["aws"]["api_endpoint"] = outputs["ApiEndpointUrl"]

    cfg["agent"]["provider"] = "bedrock"

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(" ✅ Configuration basculée en mode 'live' avec endpoints AWS réels.")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║      DÉPLOIEMENT TOTAL AWS - FRAUD DETECTION AGENTIC AI (BOTO3)       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    if not check_aws_credentials():
        sys.exit(1)

    outputs = deploy_cloudformation_stack()
    bucket = outputs.get("ArtifactsS3BucketName", "")
    if bucket:
        upload_models_to_s3(bucket)

    update_lambda_codes()
    update_project_settings(outputs)

    print("\n" + "=" * 70)
    print(" 🎉 DÉPLOIEMENT AWS TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)
    print(f" 🌐 API Gateway REST URL : {outputs.get('ApiEndpointUrl', 'N/A')}")
    print(f" 📦 Bucket S3 Modèles    : {outputs.get('ArtifactsS3BucketName', 'N/A')}")
    print(f" 📊 DynamoDB Alertes     : {outputs.get('AlertsDynamoTableName', 'N/A')}")
    print(f" 🌊 Kinesis Stream       : {outputs.get('KinesisStreamName', 'N/A')}")
    print("=" * 70)
    print("\nPour lancer le tableau de bord Streamlit connecté à AWS :")
    print("  streamlit run src/dashboard/app.py\n")


if __name__ == "__main__":
    main()
