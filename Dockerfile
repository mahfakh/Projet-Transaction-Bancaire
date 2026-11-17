# Image unique pour les 3 fonctions Lambda (scoring Kinesis, investigation, création d'alerte).
# Le template choisit le handler de chaque fonction (ImageConfig.Command).
#
# Construction (depuis la racine du projet) :
#   docker build --platform linux/amd64 --provenance=false -t fraud-detection-lambda .
# --provenance=false : Lambda refuse les images multi-manifestes générées par défaut par Docker.

FROM public.ecr.aws/lambda/python:3.13

# LightGBM a besoin de la bibliothèque OpenMP (libgomp), absente de l'image de base minimale
RUN dnf install -y libgomp && dnf clean all

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Code, configuration et modèle entraîné (le modèle tourne DANS la Lambda : pas d'endpoint SageMaker)
COPY config/ ${LAMBDA_TASK_ROOT}/config/
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY data/models/ ${LAMBDA_TASK_ROOT}/data/models/
COPY data/reports/model_metrics.json ${LAMBDA_TASK_ROOT}/data/reports/model_metrics.json

# Sur AWS, toujours en mode live (DynamoDB) ; le template peut surcharger ces valeurs
ENV APP_MODE=live \
    PYTHONDONTWRITEBYTECODE=1

CMD ["src.aws.lambda_handler.score_handler"]
