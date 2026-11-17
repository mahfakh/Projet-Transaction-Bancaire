# Déploiement complet sur AWS (AWS Academy Learner Lab), depuis la racine du projet :
#   .\infrastructure\deploy.ps1
#   .\infrastructure\deploy.ps1 -AlertEmail "vous@exemple.com"   # + e-mails pour les alertes HIGH
#   .\infrastructure\deploy.ps1 -SkipBuild                         # redéploie la pile sans reconstruire l'image
#
# Étapes : 1) construit l'image Docker  2) l'envoie UNE fois dans ECR (étiquetée avec le commit git)
#          3) déploie la pile SAM, dont les 3 fonctions Lambda partagent cette image.
param(
    [string]$Region = "us-east-1",
    [string]$Repository = "fraud-detection-lambda",
    [string]$AlertEmail = "",
    [ValidateSet("mock", "bedrock")][string]$LlmProvider = "bedrock",
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$env:SAM_CLI_TELEMETRY = "0"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Checked([scriptblock]$Command, [string]$Step) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'étape : $Step (code $LASTEXITCODE)" }
}

$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
$sam = (Get-Command sam -ErrorAction SilentlyContinue).Source
if (-not $sam) { $sam = "C:\Program Files\Amazon\AWSSAMCLI\bin\sam.cmd" }

Write-Host "== Vérification des identifiants AWS" -ForegroundColor Cyan
$account = & $aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0) { throw "Identifiants AWS invalides ou expirés : recopiez le bloc 'AWS Details' du Learner Lab dans ~/.aws/credentials" }

$registry = "$account.dkr.ecr.$Region.amazonaws.com"
$tag = (git rev-parse --short HEAD).Trim()
if (git status --porcelain) { $tag = "$tag-dirty-$(Get-Date -Format 'HHmmss')" }  # modifications non commitées
$imageUri = "$registry/${Repository}:$tag"

if (-not $SkipBuild) {
    Write-Host "== Dépôt ECR $Repository" -ForegroundColor Cyan
    & $aws ecr describe-repositories --repository-names $Repository --region $Region *> $null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked { & $aws ecr create-repository --repository-name $Repository --region $Region --image-scanning-configuration scanOnPush=true | Out-Null } "création du dépôt ECR"
    }

    Write-Host "== Construction de l'image ($tag)" -ForegroundColor Cyan
    Invoke-Checked { docker build --platform linux/amd64 --provenance=false -t "${Repository}:$tag" . } "docker build"

    Write-Host "== Envoi unique de l'image vers ECR" -ForegroundColor Cyan
    Invoke-Checked { & $aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry } "connexion à ECR"
    Invoke-Checked { docker tag "${Repository}:$tag" $imageUri } "docker tag"
    Invoke-Checked { docker push $imageUri } "docker push"
} else {
    $imageUri = & $aws lambda get-function --function-name fraud-score-prototype --region $Region --query "Code.ImageUri" --output text
    Write-Host "== Image actuelle réutilisée : $imageUri" -ForegroundColor Cyan
}

Write-Host "== Déploiement de la pile (CloudFormation via SAM)" -ForegroundColor Cyan
$overrides = "ImageUri=$imageUri Environment=prototype LlmProvider=$LlmProvider"
if ($AlertEmail) { $overrides += " AlertEmail=$AlertEmail" }
Invoke-Checked { & $sam deploy --parameter-overrides $overrides --no-confirm-changeset } "sam deploy"

Write-Host "`n== Terminé. Ressources créées :" -ForegroundColor Green
& $aws cloudformation describe-stacks --stack-name fraud-detection-prototype --region $Region --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output table
if ($AlertEmail) { Write-Host "Pensez à cliquer sur le lien de confirmation envoyé à $AlertEmail." -ForegroundColor Yellow }
