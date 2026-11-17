# Déploiement AWS (AWS Academy Learner Lab)

## Architecture déployée

```
kinesis_producer.py (PC) ──► Kinesis (1 shard) ──► Lambda fraud-score ──► DynamoDB (transactions, historique porteurs)
                                                      │  └─ événement invalide / échec répété ──► SQS fraud-dlq
                                                      └─ risque INVESTIGATE ou HIGH ──► Step Functions fraud-investigation
                                                             InvestigateTransaction (Lambda) → CreateAlert (Lambda, DynamoDB + métrique)
                                                             → si HIGH : NotifyAnalyst (SNS e-mail) → AwaitHumanReview
Tableau de bord Streamlit (APP_MODE=live) ──► lit / met à jour les alertes DynamoDB (revue humaine)
CloudWatch : logs JSON des Lambdas, métrique HighRiskAlertCount (EMF), alarme ≥ 5 alertes HIGH / 5 min
```

Spécificités AWS Academy : toutes les ressources utilisent le rôle existant **LabRole** (création de
rôles interdite) ; région **us-east-1** ; Bedrock peut être bloqué (repli déterministe signalé par
`generated_by = rule_based_fallback`).

## Coût estimé (budget Learner Lab : 50 $)

| Ressource | Coût |
|---|---|
| Kinesis, 1 shard provisionné | ~0,36 $/jour **même sans trafic** → supprimer après la soutenance |
| Lambda, DynamoDB à la demande, SQS, SNS, Step Functions | ~0 $ au volume d'une démonstration |
| ECR (image ~0,4 Go compressée), CloudWatch Logs (7 jours) | quelques centimes |

## Prérequis (une fois)

- Docker Desktop démarré, AWS CLI et AWS SAM CLI installés
- Identifiants du Learner Lab dans `~/.aws/credentials` (à recopier à chaque session : *AWS Details* → *AWS CLI*)
- Vérification : `aws sts get-caller-identity`

## Déployer

Depuis la racine du projet :

```powershell
sam build          # construit l'image Docker (template : infrastructure/template.yaml, voir samconfig.toml)
sam deploy         # affiche les ressources à créer puis demande confirmation (y)
```

Option : recevoir les alertes HIGH par e-mail (cliquer ensuite sur le lien de confirmation reçu) :

```powershell
sam deploy --parameter-overrides "Environment=prototype LlmProvider=bedrock AlertEmail=vous@exemple.com"
```

## Démonstration

```powershell
# Envoyer des transactions (normales, frauduleuses, 2 invalides, 1 attaque « carte compromise »)
.\.venv\Scripts\python.exe src\stream\kinesis_producer.py --stream banking-transactions-prototype --count 30 --fraud-ratio 0.3 --invalid 2 --attack 1

# Tableau de bord sur les données AWS réelles
$env:APP_MODE="live"; $env:ALERTS_TABLE="FraudDetectionAlerts-prototype"; $env:TRANSACTIONS_TABLE="FraudDetectionTransactions-prototype"; $env:CARD_HISTORY_TABLE="FraudDetectionCardHistory-prototype"
.\.venv\Scripts\streamlit.exe run src\dashboard\app.py
```

Dans la console AWS : *Step Functions* → `fraud-investigation-prototype` (graphe de chaque exécution),
*CloudWatch* → *Log groups* `/aws/lambda/fraud-*`, *SQS* → `fraud-dlq-prototype` (événements rejetés).

## Tout supprimer (indispensable après la soutenance)

```powershell
sam delete --stack-name fraud-detection-prototype --region us-east-1
```

## Tester l'image localement (sans AWS)

```powershell
docker build --platform linux/amd64 --provenance=false -t fraud-detection-lambda .
docker run --rm -p 9000:8080 -e APP_MODE=local -e LLM_PROVIDER=mock fraud-detection-lambda src.aws.lambda_handler.investigate_handler
# puis POST d'un événement sur http://localhost:9000/2015-03-31/functions/function/invocations
```
