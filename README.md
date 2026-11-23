# Assistant de segmentation client

Projet d'IA (BUT 3), réalisé en binôme par **Martin Cuinet** et **Alexis Bollengier**.

Sujet n° 3 : **Clustering × Tool use**. La clientèle d'un opérateur télécom est découpée en groupes par un clustering, puis un assistant (un LLM) répond aux questions d'un responsable marketing en appelant des fonctions Python qui interrogent ces groupes.

Exemples de questions :
- « Décris le groupe 2. »
- « Compare les groupes 1 et 3. »
- « Dans quel groupe est le client 7590-VHVEG ? »
- « Un client depuis 3 mois paie 95 $ par mois : dans quel groupe tombe-t-il ? »

## Fonctionnement

```
Question de l'utilisateur
      │
      ▼
LLM : choisit un outil et ses arguments   (ex. describe_group(group_id=2))
      │
      ▼
Notre code Python exécute l'outil sur les données du clustering
      │
      ▼
LLM : lit le résultat et rédige la réponse en français
```

Les deux notions sont indispensables :
- **sans le clustering**, les outils n'ont pas de groupes à décrire ;
- **sans le tool use**, on ne peut pas interroger les groupes en langage naturel, et le LLM seul inventerait les chiffres.

## Installation et lancement

1. Créer un environnement Python et installer les dépendances :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Créer un fichier `.env` à la racine du projet, avec les trois lignes du fournisseur de LLM (comme dans les TP6 et TP7). Api gratuite : Groq :
   ```
   LLM_BASE_URL=https://api.groq.com/openai/v1
   LLM_API_KEY=votre-clé
   LLM_MODEL=openai/gpt-oss-20b
   ```
   Le fichier `.env` est exclu du dépôt par le `.gitignore` : aucune clé d'API n'est publiée.
3. Lancer l'application :
   ```bash
   streamlit run app.py
   ```
   La page s'ouvre sur http://localhost:8501.

Le modèle de clustering est déjà entraîné et enregistré dans `models/`. Pour le réentraîner, exécuter le notebook `notebooks/clustering_github.ipynb`.

## Organisation du dépôt

| Fichier | Rôle |
|---|---|
| `notebooks/clustering_github.ipynb` | Clustering : nettoyage, standardisation, choix de k, K-Means, profils des groupes, sauvegarde |
| `data/telco_churn.csv` | Données d'origine |
| `data/telco_clusters.csv` | Données avec le groupe de chaque client (produit par le notebook) |
| `models/scaler.joblib`, `models/kmeans.joblib` | Modèle entraîné (produit par le notebook) |
| `src/tools.py` | Les 4 outils que le LLM peut appeler |
| `src/llm_integration.py` | Connexion au LLM, description des outils, boucle de l'assistant (reprise du TP7) |
| `app.py` | Interface web Streamlit |
| `TP1_Correction.ipynb` | Correction du TP1 de clustering, qui a servi de base au notebook |

## Données

- **Jeu de données** : *Telco Customer Churn*, IBM, utilisé dans le TP2.
- **Licence** : Apache 2.0.
- **Contenu** : 7 043 clients d'un opérateur télécom, avec leur ancienneté (`tenure`, en mois), leur facture mensuelle (`MonthlyCharges`), leur contrat, leurs services et s'ils ont résilié (`Churn`).
- **Nettoyage** : la colonne `TotalCharges` est lue comme du texte et contient 11 valeurs vides (clients avec une ancienneté de 0 mois) ; elle est convertie en nombre et les valeurs vides sont remplacées par 0.

## Notion 1 : le clustering

- **Méthode** : K-Means, sur deux variables, l'ancienneté (`tenure`) et la facture mensuelle (`MonthlyCharges`), standardisées avec un `StandardScaler`. La démarche reprend celle du TP1.
- **Choix du nombre de groupes** : méthode du coude et score de silhouette, pour k de 2 à 10.

| k | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0,411 | 0,446 | **0,476** | 0,430 | 0,415 | 0,434 | 0,433 | 0,440 | 0,430 |

La silhouette est maximale pour **k = 4** : nous retenons 4 groupes.

- **Les groupes obtenus** (le taux de départ n'a pas servi au clustering, il sert à décrire les groupes) :

| Groupe | Nom | Clients | Ancienneté moyenne | Facture moyenne | Taux de départ |
|---|---|---|---|---|---|
| 0 | Nouveaux petits forfaits | 1 732 | 10,5 mois | 32,47 $ | 25 % |
| 1 | Fidèles gros consommateurs | 1 952 | 58,8 mois | 93,02 $ | 16 % |
| 2 | Nouveaux gros forfaits à risque | 2 204 | 14,8 mois | 81,22 $ | 49 % |
| 3 | Fidèles économes | 1 155 | 54,1 mois | 34,03 $ | 5 % |

## Notion 2 : le tool use

Le LLM ne lit pas directement les données : il dispose de 4 outils, décrits dans `src/llm_integration.py`, et décide lui-même lequel appeler.

| Outil | Ce qu'il fait |
|---|---|
| `describe_group(group_id)` | Profil d'un groupe : nom, taille, ancienneté et facture moyennes, taux de départ, contrat et internet les plus fréquents, part de seniors |
| `compare_groups(group_a, group_b)` | Profils de deux groupes et écarts entre eux |
| `client_details(customer_id)` | Informations d'un client existant, son groupe et le profil de ce groupe |
| `predict_group(tenure, monthly_charges)` | Groupe d'un nouveau client, prédit par le modèle K-Means, et profil de ce groupe |

La boucle de l'assistant reprend celle du TP7 : le LLM demande un outil, notre code l'exécute, le résultat lui est renvoyé, et ainsi de suite jusqu'à sa réponse finale (5 étapes au maximum). Dans l'application, l'encadré « 🔧 Outils appelés » sous chaque réponse montre les outils appelés, leurs arguments et leurs résultats.

## Expériences

### Clustering
Comparaison des valeurs de k de 2 à 10 avec le score de silhouette (tableau ci-dessus) : k = 4 obtient le meilleur score (0,476).

### Tool use
*À faire* : évaluer le choix des outils sur un jeu de questions annotées à la main, comme dans le TP7.

## Limites constatées

- **Le LLM peut mal interpréter des résultats justes.** En comparant les groupes 1 et 3, il a écrit « taux de départ 10,8 % supérieur » alors qu'il s'agit d'un écart de 10,8 points (15,6 % contre 4,8 %). Il a aussi affirmé que le groupe 3 était « plus jeune », alors que les données ne contiennent pas l'âge.
- **Le LLM a confondu un client et son groupe.** Pour le client 7590-VHVEG, il a présenté l'ancienneté de ce client (1 mois) comme l'ancienneté moyenne du groupe (10,5 mois en réalité). Correction : `client_details` renvoie maintenant aussi le profil du groupe. Nous avons également ajouté au message système une consigne demandant de ne tirer aucune conclusion sans donnée et de ne pas confondre client et groupe.
- **Le clustering n'utilise que deux variables** (ancienneté et facture). Les autres informations (contrat, services…) ne servent qu'à décrire les groupes.
- **Les noms des groupes sont écrits à la main** dans `src/tools.py` d'après leurs numéros : si le clustering est relancé avec d'autres réglages, les numéros peuvent changer et les noms doivent être vérifiés.
