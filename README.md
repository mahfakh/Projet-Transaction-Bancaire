# Agentic AI for Real-Time Banking Fraud Detection on AWS

> **Prototype académique de démonstration** : Ce système analyse des flux de transactions, détecte des comportements inhabituels et assiste un analyste humain via un agent LLM explicable. Il ne constitue pas un système bancaire de production et n'effectue aucun blocage automatique réel.

---

## 1. Architecture Générale

```
[ Générateur temps réel / Dataset IEEE-CIS ]
                   │
                   ▼
       [ Ingestion & Validation (F1) ] ── (Rejet) ──► [ Dead-Letter Queue (DLQ) ]
                   │
                   ▼
     [ Contexte Comportemental (F4) ] (Vélocité 24h, Écart aux habitudes, Changement terminal)
                   │
                   ▼
      [ Modèle ML LightGBM (F3) ] (Score de risque 0.0 - 1.0, ROC-AUC: 0.929, PR-AUC: 0.679)
                   │
         Score >= Seuil d'investigation ?
          ┌────────┴────────┐
         OUI               NON
          │                 │
          ▼                 │
[ Agent LLM d'Investigation (F5) ] │
(Facteurs observés, Manques, Incertitude)
          │                 │
          ▼                 ▼
[ File d'Alertes DynamoDB / S3 ] ◄── [ Persistance Transactions (F8) ]
          │
          ▼
[ Tableau de Bord Streamlit (F7) ]
          │
          ▼
[ Revue Humaine & Décision Analyste (F6) ] ──► [ Journal d'Audit CloudWatch (F9) ]
```

---

## 2. Performances du Modèle de Détection (F3) & Résultats des Expérimentations

Évaluation sur un jeu de test de validation indépendant (**stratifié avec fort déséquilibre de classes** ~2.5% de fraudes sur 150 000 transactions) :

### Comparatif des Cibles Expérimentales Recommandées

| Critère / Métrique | Objectif Recommandé | Valeur Obtenue | Statut | Commentaire Métier |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | **> 0.93** | **0.9430** | **ATTEINT ✅** | Pouvoir séparateur global exceptionnel |
| **PR-AUC (Average Precision)** | **> 0.70** | **0.7340** | **ATTEINT ✅** | Métrique reine sur distribution fortement asymétrique |
| **Rappel (Seuil Métier 0.20)** | **> 70 %** | **72.17 %** | **ATTEINT ✅** | **72+ % des fraudes réelles interceptées** |
| **F1-Score (Seuil 0.35)** | **> 0.70** | **0.7052** | **ATTEINT ✅** | Compromis mathématique optimal Precision / Recall |

### Matrice d'Arbitrage Multi-Seuils (Coût Métier Bancaire)

> **Principe économique :** En fraude bancaire, le coût d'un **Faux Négatif** (perte financière directe et litige) est **8 à 15 fois supérieur** au coût de traitement d'un **Faux Positif** (simple appel ou SMS de vérification).

| Seuil | Rappel (Fraudes Détectées) | Précision | F1-Score | Fraudes Manquées (FN) | Fausses Alertes (FP) | Profil d'Usage |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.20** | **72.2 %** | **57.5 %** | **0.6402** | **285** | 412 | **Seuil d'alerte recommandé (Sécurité maximale)** |
| **0.25** | **69.3 %** | **65.2 %** | **0.6716** | 315 | 287 | Seuil équilibré flux soutenu |
| **0.35** | **65.4 %** | **76.6 %** | **0.7052** | 355 | 155 | **F1 maximal théorique** |
| **0.50** | **58.8 %** | **85.2 %** | **0.6960** | 422 | 79 | Seuil conservateur (Trop de fraudes manquées) |

### Enseignements des Expérimentations
1. **SMOTE** : Conformément aux mises en garde, SMOTE produit une dégradation légère (ROC-AUC à 0.9201, PR-AUC à 0.654) due à l'interpolation synthétique dans un espace creux de 70+ variables.
2. **`scale_pos_weight = 3.5`** : Bien supérieur à SMOTE, il pondère le gradient lors de la construction des arbres sans déformer la structure des données.
3. **Feature Engineering Comportemental** : Les ratios par porteur (`amt_to_card1_mean_ratio`), la décomposition temporelle de `TransactionDT` et les centimes ont augmenté le PR-AUC de **+0.055**.

---

## 3. Structure du Projet

```plaintext
aws/
├── config/
│   └── settings.yaml                      # Configuration des seuils, hyperparamètres et AWS
├── data/
│   ├── models/                            # Modèle sérialisé (.joblib), feature engineer et préprocesseur
│   ├── reports/                           # Métriques JSON et rapports d'évaluation EDA / benchmarks
│   └── storage/                           # Persistance locale (Mock DynamoDB/S3 sans frais)
├── ieee-fraud-detection/                  # Dataset source IEEE-CIS (train/test transactions & identity)
├── infrastructure/
│   ├── cloudformation_template.yaml       # Template IaC (Kinesis, DynamoDB, S3, SNS, CloudWatch)
│   └── step_functions_definition.json     # Machine d'état Step Functions (Orchestration AWS)
├── src/
│   ├── agent/
│   │   ├── llm_investigator.py            # Agent LLM avec contraintes strictes et sortie structurée
│   │   └── schemas.py                     # Schémas et contrats Pydantic (Section 4)
│   ├── aws/
│   │   ├── adapter.py                     # Double adaptateur : Local Mock ↔ AWS Boto3
│   │   └── orchestration.py               # Pipeline d'orchestration end-to-end
│   ├── behavioral/
│   │   └── context_engine.py              # Moteur d'historique glissant et signaux d'anomalie
│   ├── dashboard/
│   │   └── app.py                         # Interface interactive Streamlit (Supervision & Revue)
│   ├── data/
│   │   ├── explorer.py                    # Exploration statistique du dataset (EDA)
│   │   ├── feature_engineer.py            # Ingénierie des caractéristiques comportementales
│   │   └── preprocessor.py                # Pipeline de prétraitement sans fuite de cible
│   ├── models/
│   │   ├── experiment.py                  # Étude comparative : SMOTE, scale_pos_weight, seuils
│   │   ├── scorer.py                      # Service d'inférence temps réel et seuillage
│   │   └── train.py                       # Entraînement et évaluation LightGBM
│   └── stream/
│       ├── generator.py                   # Ingestion temps réel et file de rejet (DLQ)
│       └── seed_demo.py                   # Injection de transactions de test
├── tests/
│   └── test_fraud_system.py               # Tests unitaires et d'intégration pytest
├── requirements.txt
└── README.md
```

---

## 4. Guide d'Exécution & Démarrage Rapide

### A. Prérequis et Activation de l'Environnement Virtuel

Si vous utilisez un environnement virtuel `.venv`, activez-le avant d'exécuter les scripts :

- **PowerShell (Windows)** :
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(Si l'exécution de scripts est restreinte dans PowerShell : `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*
- **Invite de commandes (cmd.exe)** :
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Bash / Linux / macOS** :
  ```bash
  source .venv/bin/activate
  ```

Installation / mise à jour des dépendances :
```bash
pip install -r requirements.txt
```

---

### B. Validation par les Tests Unitaires & d'Intégration
Pour vérifier le bon fonctionnement de tous les modules (ingestion, DLQ, absence de fuite de cible, inférence ML, détection d'anomalies comportementales, agent LLM et audit) :
```bash
pytest tests/test_fraud_system.py
```

---

### C. Exploration des Données (EDA - F2)
Pour analyser la distribution des transactions, le taux de fraude réel (~2.5 %) et les jointures Identity :
```bash
python src/data/explorer.py
```

---

### D. Benchmark & Expérimentations d'Optimisation ML
Pour reproduire la comparaison (Feature Engineering, SMOTE vs `scale_pos_weight`, analyse coût métier bancaire et recherche de seuils optimaux) :
```bash
python src/models/experiment.py
```

---

### E. Entraînement du Modèle Champion (F3)
Pour réentraîner le modèle LightGBM optimisé et régénérer les artefacts sérialisés dans `data/models/` :
```bash
python src/models/train.py
```

---

### F. Injection de Données de Démonstration (Flux & Alertes)
Pour alimenter la base de données locale avec un lot initial de transactions simulées et d'alertes à réviser :
```bash
python src/stream/seed_demo.py
```

---

### G. Lancement du Tableau de Bord Streamlit (F7)
Pour démarrer l'interface web de supervision et de revue analyste :

```bash
streamlit run src/dashboard/app.py
```

Ou directement avec l'interpréteur de votre environnement virtuel :
```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py
```

Le tableau de bord s'ouvrira automatiquement à l'adresse locale : `http://localhost:8501`.

---

## 5. Garde-fous et Éthique de l'IA (Human-in-the-Loop)

1. **Aucun blocage unilatéral** : L'agent LLM ne prend aucune décision bancaire finale. Il produit une explication factuelle structurée pour éclairer l'analyste.
2. **Pas d'hallucination de données** : Le prompt contraint interdit à l'agent d'extrapoler des faits non présents dans les signaux comportementaux ou la transaction.
3. **Auditabilité complète** : Chaque décision de l'analyste (statut, note explicative) est horodatée et consignée dans le journal d'audit (`audit_log.jsonl` / CloudWatch).
4. **Préservation de la confidentialité (PII)** : Seules les variables synthétisées et pseudonymisées sont transmises à l'agent IA.
