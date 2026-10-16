import streamlit as st
import ollama
from src.rag_assistant.retrieval import Retriever
from src.rag_assistant.ingest import IngestionPipeline

st.set_page_config(page_title="Assistant Service-Public", layout="wide")

@st.cache_resource
def load_rag():
    pipeline = IngestionPipeline(persist_directory="/content/chroma_db_final")
    return Retriever(ingestion_pipeline=pipeline)

retriever = load_rag()

st.title("🏛️ Assistant Service-Public.gouv.fr")
st.markdown("Posez vos questions sur le logement, les papiers d'identité, etc.")

query = st.text_input("Votre question :")

if st.button("Rechercher") and query:
    with st.spinner("Recherche des documents pertinents..."):
        chunks = retriever.retrieve_and_rerank(query, top_k=4)

    if not chunks:
        st.warning("Aucun document trouvé pour répondre à cette question.")
    else:
        st.subheader("Sources retrouvées :")
        for i, chunk in enumerate(chunks, 1):
            st.info(f"**[{i}] {chunk['title']}** (MàJ: {chunk['date']})\n\n_{chunk['body']}_")

        with st.spinner("Génération de la réponse..."):
            sources_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
            system_prompt = "Tu es un assistant administratif. Réponds en citant les sources entre crochets (ex: [1])."
            user_prompt = f"Sources:\n{sources_text}\n\nQuestion: {query}"

            response = ollama.chat(
                model="llama3.1:8b",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )

        st.subheader("Réponse de l'Assistant :")
        st.write(response['message']['content'])
