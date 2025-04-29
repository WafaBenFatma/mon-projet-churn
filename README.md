
#  Projet de Prédiction du Churn Client

##  Objectif

Prédire le churn (désabonnement) des clients à partir de leurs données historiques dans le secteur des télécommunications, afin d’optimiser les campagnes de rétention.

---

##  Architecture Globale du Projet

```mermaid
graph TD
 B[Préparation des Données]
    B --> C[Analyse Exploratoire]
    C --> D[Modélisation]
    D --> E[Évaluation]
    E --> F[Déploiement]
    
    subgraph "Détails Techniques"
        B --> B1[Nettoyage]
        B --> B2[Feature Engineering]
        B --> B3[Balancing]
        D --> D1[Regression Logistique]
        D --> D2[Random Forest]
        D --> D3[XGBoost]
        E --> E1[Validation Croisée]
        E --> E2[Métriques]
    end

```

---

##  Données

###  Sources
- Données historiques internes d’un opérateur télécom
- Données clients : démographie, services, facturation, utilisation

###  Répartition des Variables

```mermaid
pie title Répartition des Variables
    "Démographiques" : 35
    "Services" : 45
    "Facturation" : 20
```

---

##  Préparation des Données

```mermaid
flowchart LR
    A[Données Brutes] --> B{Qualité}
    B -->|Manquantes| C[Imputation]
    B -->|Dupliquées| D[Suppression]
    B -->|Incohérentes| E[Correction]
```

- Imputation des valeurs manquantes
- Suppression des doublons
- Correction des incohérences
- Standardisation et encodage des variables

---

##  Rééquilibrage des Classes

```mermaid
pie title Distribution après équilibrage
    "SMOTE - Classe 0" : 50
    "SMOTE - Classe 1" : 50
    "ADASYN - Classe 0" : 50
    "ADASYN - Classe 1" : 50

```

- **SMOTE :** Génère des exemples synthétiques entre les plus proches voisins de la classe minoritaire.
- **ADASYN :** Génère plus d'exemples pour les observations difficiles à apprendre.


**Objectif :** Améliorer la capacité des modèles à détecter le churn.
---

##  Analyse Exploratoire

```mermaid
graph LR
    A[Âge] -->|Corrélation| B[Churn]
    C[Forfait] -->|Impact| B
    D[Durée Appels] -->|Relation| B
```

- Identification de variables fortement liées au churn
- Détection de corrélations utiles pour la modélisation

---

##  Modélisation

```mermaid
classDiagram
    class Modèles{
        +Régression Logistique
        +Random Forest
        +XGBoost
        +Gradient Boosting
        +train()
        +predict()
    }
```

- Comparaison de plusieurs modèles supervisés
- Focus sur les algorithmes robustes aux données déséquilibrées

---

## Comparaison des Performances : SMOTE vs ADASYN
```mermaid
gantt
    title F1-Score selon la Technique de Rééquilibrage
    dateFormat  X
    axisFormat %s

    section SMOTE
    XGBoost : 0, 82
    Random Forest : 0, 80
    Logistique : 0, 75

    section ADASYN
    XGBoost : 0, 81
    Random Forest : 0, 78
    Logistique : 0, 74

``` 
- **SMOTE + XGBoost =** meilleures performances

- **ADASYN** légèrement moins performant sur tous les modèles
##  Évaluation des Modèles

```mermaid
gantt
    title Comparaison des Métriques (AUC-ROC)
    dateFormat  X
    axisFormat %s

    section AUC-ROC
    XGBoost : 0, 87
    Random Forest : 0, 85
    Regression Logistique : 0, 78

```

- Évaluation par **Validation Croisée**
- Meilleur compromis performance/interprétabilité avec **XGBoost**

---

##  Déploiement

```mermaid
sequenceDiagram
    Utilisateur->>+UI: Upload données
    UI->>+API: Requête /predict
    API->>+Modèle: Prédiction
    Modèle-->>-API: Résultats
    API-->>-UI: Réponse JSON
    UI-->>-Utilisateur: Affichage
```

- Création d’une **API de prédiction** accessible par une interface utilisateur
- Intégration dans les outils décisionnels de l'entreprise

---

## Conclusion
- Le projet a permis de concevoir un pipeline complet de prédiction du churn.

- **SMOTE** combiné à **XGBoost** a donné les meilleurs résultats.
