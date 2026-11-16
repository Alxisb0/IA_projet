from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent

CLIENTS = pd.read_csv(ROOT / "data/clients_clusters.csv")
MODELE = joblib.load(ROOT / "models/clustering.joblib")

GROUPES = sorted(CLIENTS["cluster"].unique().tolist())


def describe_group(group_id: int) -> dict:
    if group_id not in GROUPES:
        return {"error": f"Le groupe {group_id} n'existe pas. Groupes valides : {GROUPES}"}

    groupe = CLIENTS[CLIENTS["cluster"] == group_id]

    return {
        "groupe": group_id,
        "nombre_clients": len(groupe),
        "part_de_la_clientele": round(len(groupe) / len(CLIENTS), 3),
        "anciennete_moyenne_mois": round(float(groupe["tenure"].mean()), 1),
        "facture_mensuelle_moyenne": round(float(groupe["MonthlyCharges"].mean()), 2),
        "taux_de_depart": round(float((groupe["Churn"] == "Yes").mean()), 3),
        "contrat_le_plus_frequent": groupe["Contract"].mode()[0],
        "internet_le_plus_frequent": groupe["InternetService"].mode()[0],
        "part_seniors": round(float(groupe["SeniorCitizen"].mean()), 3),
    }


if __name__ == "__main__":
    print(describe_group(2))
    print(describe_group(12))
