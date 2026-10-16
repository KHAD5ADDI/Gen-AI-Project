import time
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from .generation import ROLE, RULES, Reponse, format_passages, generate, parse_and_validate
from .tracing import Trace, chat

AGENT_PROMPT = (
    ROLE + " "
    "Tu disposes d'outils pour chercher dans ces fiches : utilise search_corpus avant de répondre, "
    "plusieurs fois avec des requêtes différentes si la question a plusieurs parties, et "
    "get_document pour lire une fiche en entier si un passage ne suffit pas. "
    "Quand tu as assez d'éléments, réponds sans appeler d'outil.\n\n" + RULES
)

def run_agent(llm, retriever, question: str, trace: Trace, max_steps: int = 3):
    seen = {}
    opened = set()

    def search_corpus(query: str) -> str:
        chunks = retriever.search(query)
        for c in chunks:
            seen.setdefault(c["doc_id"], c)
        return format_passages(chunks) or "Aucun passage trouvé."

    def get_document(doc_id: str) -> str:
        text = retriever.get_document(doc_id)
        if text is None: return f"Document inconnu : {doc_id}"
        opened.add(doc_id)
        return f'<document doc_id="{doc_id}">\n{text}\n</document>'

    tools = [FunctionTool.from_defaults(fn=search_corpus), FunctionTool.from_defaults(fn=get_document)]
    by_name = {t.metadata.name: t for t in tools}
    messages = [
        ChatMessage(role="system", content=AGENT_PROMPT),
        ChatMessage(role="user", content=f"<question>{question}</question>"),
    ]

    final_text = None
    for step in range(max_steps):
        response = chat(llm, messages, trace, purpose=f"agent_step_{step}", tools=tools)
        calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        if not calls:
            final_text = response.message.content
            break
        messages.append(response.message)
        for call in calls:
            t0 = time.perf_counter()
            tool = by_name.get(call.tool_name)
            output = tool.call(**call.tool_kwargs).content if tool else f"Outil inconnu : {call.tool_name}"
            trace.record_tool(call.tool_name, call.tool_kwargs, time.perf_counter() - t0, len(output))
            messages.append(ChatMessage(role="tool", content=output, additional_kwargs={"tool_call_id": call.tool_id, "name": call.tool_name}))

    retrieved_ids = list(seen) + [d for d in opened if d not in seen]
    allowed = set(retrieved_ids)
    
    if final_text is not None:
        try:
            return parse_and_validate(final_text, allowed), retrieved_ids
        except ValueError as err:
            trace.errors.append(f"agent_final: {err}")
    else:
        trace.errors.append("agent: max_steps atteint")

    # Repli RAG classique
    chunks = list(seen.values()) or retriever.search(question)
    if not seen:
        retrieved_ids = [c["doc_id"] for c in chunks]
    reponse = generate(llm, question, chunks, trace)
    return reponse, retrieved_ids
