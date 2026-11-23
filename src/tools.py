from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent

CLIENTS = pd.read_csv(ROOT / "data/telco_clusters.csv")
SCALER = joblib.load(ROOT / "models/scaler.joblib")
KMEANS = joblib.load(ROOT / "models/kmeans.joblib")

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


def compare_groups(group_a: int, group_b: int) -> dict:
    a = describe_group(group_a)
    b = describe_group(group_b)
    if "error" in a:
        return a
    if "error" in b:
        return b

    ecarts = {}
    for cle in ["anciennete_moyenne_mois", "facture_mensuelle_moyenne", "taux_de_depart", "part_seniors"]:
        ecarts[cle] = round(a[cle] - b[cle], 3)

    return {
        f"groupe_{group_a}": a,
        f"groupe_{group_b}": b,
        f"ecarts_{group_a}_moins_{group_b}": ecarts,
    }


def client_details(customer_id: str) -> dict:
    lignes = CLIENTS[CLIENTS["customerID"] == customer_id]
    if lignes.empty:
        return {"error": f"Aucun client avec l'identifiant {customer_id}"}

    client = lignes.iloc[0]

    return {
        "customer_id": customer_id,
        "groupe": int(client["cluster"]),
        "anciennete_mois": int(client["tenure"]),
        "facture_mensuelle": float(client["MonthlyCharges"]),
        "total_facture": float(client["TotalCharges"]),
        "contrat": client["Contract"],
        "internet": client["InternetService"],
        "senior": bool(client["SeniorCitizen"]),
        "est_parti": client["Churn"] == "Yes",
    }


def predict_group(tenure: int, monthly_charges: float) -> dict:
    if tenure < 0 or monthly_charges < 0:
        return {"error": "L'ancienneté et la facture doivent être positives"}

    nouveau = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly_charges}])
    groupe = int(KMEANS.predict(SCALER.transform(nouveau))[0])

    return {
        "anciennete_mois": tenure,
        "facture_mensuelle": monthly_charges,
        "groupe_predit": groupe,
        "profil_du_groupe": describe_group(groupe),
    }


if __name__ == "__main__":
    print(client_details("7590-VHVEG"))
    print(client_details("XXXX"))
    print(predict_group(3, 95.0))
    print(predict_group(-1, 50))
