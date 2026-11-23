import json
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from llm_integration import assistant
from tools import NOMS

st.set_page_config(page_title="Assistant segmentation client", page_icon="📊")
st.title("Assistant segmentation client")
st.caption("Clientèle Telco découpée en 4 groupes par K-Means (ancienneté et facture mensuelle).")

with st.sidebar:
    st.header("Les groupes")
    for numero, nom in NOMS.items():
        st.write(f"**{numero}** — {nom}")
    st.header("Exemples de questions")
    st.write("- Décris le groupe 2.\n"
             "- Compare les groupes 1 et 3.\n"
             "- Dans quel groupe est le client 7590-VHVEG ?\n"
             "- Un client depuis 3 mois paie 95 $ : dans quel groupe tombe-t-il ?")
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []


def afficher_appels(appels):
    if not appels:
        return
    with st.expander(f"🔧 Outils appelés ({len(appels)})"):
        for appel in appels:
            st.markdown(f"**{appel['outil']}** `{appel['arguments']}`")
            try:
                st.json(json.loads(appel["resultat"]))
            except json.JSONDecodeError:
                st.write(appel["resultat"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        afficher_appels(message.get("appels"))

question = st.chat_input("Posez une question sur la clientèle")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    historique = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            reponse, appels = assistant(question, historique)
        st.markdown(reponse)
        afficher_appels(appels)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": reponse, "appels": appels})
