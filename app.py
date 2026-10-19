import streamlit as st
import time
import os
from src.rag_assistant.ingest import IngestionPipeline
from src.rag_assistant.retrieval import HybridRetriever
from src.rag_assistant.agent import run_agent
from src.rag_assistant.tracing import Trace

# Pour exécution locale :
from src.rag_assistant.llm import get_llm
# Pour exécution Cloud (Hugging Face Spaces) - DÉCOMMENTER :
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

st.set_page_config(page_title="Assistant Service-Public", layout="wide")

@st.cache_resource
def init_system():
    pipeline = IngestionPipeline(persist_directory="./chroma_db_final")
    retriever = HybridRetriever(ingestion_pipeline=pipeline)
    
    # Configuration LOCAL :
    llm = get_llm()
    
    # Configuration CLOUD (Hugging Face) :
    # hf_token = os.environ.get("HF_TOKEN")
    # llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
    
    return retriever, llm

retriever, llm = init_system()

st.title("🏛️ Assistant Service-Public (Agent RAG)")

query = st.chat_input("Votre question :")

if query:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        with st.spinner("Recherche et réflexion en cours..."):
            trace = Trace()
            start_time = time.perf_counter()
            reponse, retrieved_ids = run_agent(llm, retriever, query, trace)
            latency = time.perf_counter() - start_time

        if reponse.answerable:
            st.write(reponse.answer)
            if reponse.citations:
                st.caption(f"📚 **Sources :** {', '.join(reponse.citations)}")
        else:
            st.warning(reponse.answer)
            
        with st.expander("⏱️ Métriques & Traces"):
            st.write(f"**Latence:** {latency:.2f}s")
            st.write(f"**Appels d'outils:** {len(trace.tool_calls)}")
            st.write(f"**Docs consultés:** {retrieved_ids}")
