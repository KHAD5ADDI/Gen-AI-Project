import time
from llama_index.core.llms import ChatMessage


class Trace:
    def __init__(self):
        self.tool_calls = []
        self.errors = []
        self.llm_calls = []
        self.retrieved_passages = []

    def record_tool(self, name, kwargs, duration, output_len):
        self.tool_calls.append({
            "name": name,
            "kwargs": kwargs,
            "duration": round(duration, 3),
            "output_len": output_len,
        })

    def record_llm(self, purpose, duration, input_tokens, output_tokens, stop_reason):
        self.llm_calls.append({
            "purpose": purpose,
            "duration": round(duration, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "stop_reason": stop_reason,
        })

    def record_passages(self, chunks):
        self.retrieved_passages = [
            {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "title": c.get("title", "")}
            for c in chunks
        ]

    @property
    def total_tokens(self):
        return sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in self.llm_calls)

    @property
    def total_llm_latency(self):
        return round(sum(c.get("duration", 0) for c in self.llm_calls), 3)


def chat(llm, messages, trace, purpose="default", tools=None):
    t0 = time.perf_counter()
    if tools:
        response = llm.chat_with_tools(tools, chat_history=messages)
    else:
        response = llm.chat(messages)
    duration = time.perf_counter() - t0

    raw = getattr(response, "raw", {}) or {}
    usage = raw.get("usage", {}) or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    stop_reason = raw.get("stop_reason") or raw.get("finish_reason", "unknown")

    trace.record_llm(purpose, duration, input_tokens, output_tokens, stop_reason)
    return response
