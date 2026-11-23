import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from tools import describe_group, compare_groups, client_details, predict_group

load_dotenv()
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODELE = os.environ["LLM_MODEL"]


def appeler(messages, modele=None, **parametres):
    for essai in range(7):
        try:
            return client.chat.completions.create(model=modele or MODELE, messages=messages, **parametres)
        except RateLimitError:
            attente = 2 ** essai
            print(f"Trop de requêtes : nouvel essai dans {attente} s")
            time.sleep(attente)
    raise RuntimeError("Limite de débit toujours dépassée : réessayez dans une minute.")


def message_vers_dict(message):
    d = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        d["tool_calls"] = [appel.model_dump() for appel in message.tool_calls]
    return d


OUTIL_DESCRIBE = {
    "type": "function",
    "function": {
        "name": "describe_group",
        "description": ("Décrit un groupe (segment) de clients : nom, nombre de clients, ancienneté moyenne, "
                        "facture mensuelle moyenne, taux de départ, contrat et type d'internet les plus fréquents, "
                        "part de seniors."),
        "parameters": {
            "type": "object",
            "properties": {
                "group_id": {"type": "integer", "description": "numéro du groupe, de 0 à 3"},
            },
            "required": ["group_id"],
        },
    },
}

OUTIL_COMPARE = {
    "type": "function",
    "function": {
        "name": "compare_groups",
        "description": ("Compare deux groupes de clients : renvoie le profil de chacun et les écarts "
                        "(groupe A moins groupe B) sur l'ancienneté, la facture, le taux de départ et la part de seniors."),
        "parameters": {
            "type": "object",
            "properties": {
                "group_a": {"type": "integer", "description": "numéro du premier groupe, de 0 à 3"},
                "group_b": {"type": "integer", "description": "numéro du second groupe, de 0 à 3"},
            },
            "required": ["group_a", "group_b"],
        },
    },
}

OUTIL_CLIENT = {
    "type": "function",
    "function": {
        "name": "client_details",
        "description": ("Donne le groupe et les informations d'un client existant à partir de son identifiant : "
                        "ancienneté, facture, contrat, internet, senior, et s'il est parti."),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "identifiant du client, par exemple 7590-VHVEG"},
            },
            "required": ["customer_id"],
        },
    },
}

OUTIL_PREDICT = {
    "type": "function",
    "function": {
        "name": "predict_group",
        "description": ("Prédit le groupe d'un nouveau client à partir de son ancienneté et de sa facture mensuelle, "
                        "avec le modèle de clustering (K-Means), et renvoie le profil de ce groupe."),
        "parameters": {
            "type": "object",
            "properties": {
                "tenure": {"type": "integer", "description": "ancienneté du client, en mois (0 pour un nouveau client)"},
                "monthly_charges": {"type": "number", "description": "facture mensuelle du client, en dollars"},
            },
            "required": ["tenure", "monthly_charges"],
        },
    },
}

FONCTIONS = {
    "describe_group": describe_group,
    "compare_groups": compare_groups,
    "client_details": client_details,
    "predict_group": predict_group,
}
OUTILS = [OUTIL_DESCRIBE, OUTIL_COMPARE, OUTIL_CLIENT, OUTIL_PREDICT]
SYSTEME = ("Tu es un assistant pour un responsable marketing d'un opérateur télécom. La clientèle est découpée "
           "en 4 groupes (0 à 3) par un clustering. Tu réponds en français, en quelques phrases, "
           "et tu vouvoies l'utilisateur. "
           "Utilise les outils dès qu'ils sont utiles : n'invente jamais un chiffre sur les clients ou les groupes.")


def executer(appel, fonctions):
    try:
        arguments = json.loads(appel.function.arguments or "{}")
        resultat = fonctions[appel.function.name](**arguments)
        return resultat if isinstance(resultat, str) else json.dumps(resultat, ensure_ascii=False)
    except Exception as erreur:
        return f"Erreur : {erreur}"


def assistant(question, historique=None, outils=OUTILS, fonctions=FONCTIONS, systeme=SYSTEME, max_etapes=5):
    messages = [{"role": "system", "content": systeme}] + (historique or []) + [{"role": "user", "content": question}]
    appels = []
    for etape in range(max_etapes):
        message = appeler(messages, tools=outils, temperature=0).choices[0].message
        if not message.tool_calls:
            return message.content, appels
        messages.append(message_vers_dict(message))
        for appel in message.tool_calls:
            resultat = executer(appel, fonctions)
            appels.append({"outil": appel.function.name, "arguments": appel.function.arguments, "resultat": resultat})
            print(f"  🔧 {appel.function.name}({appel.function.arguments}) → {resultat[:150]}")
            messages.append({"role": "tool", "tool_call_id": appel.id, "content": resultat})
    return "Nombre maximal d'étapes atteint.", appels


if __name__ == "__main__":
    for question in ["Décris le groupe 2.",
                     "Compare les groupes 1 et 3.",
                     "Dans quel groupe est le client 7590-VHVEG ?",
                     "Un client depuis 3 mois paie 95 $ par mois : dans quel groupe tombe-t-il ?"]:
        print("❓", question)
        reponse, appels = assistant(question)
        print("💬", reponse, "\n")
