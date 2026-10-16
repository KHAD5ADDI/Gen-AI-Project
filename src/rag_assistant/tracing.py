from llama_index.core.llms import ChatMessage

class Trace:
    def __init__(self):
        self.tool_calls = []
        self.errors = []

    def record_tool(self, name, kwargs, duration, output_len):
        self.tool_calls.append({"name": name, "kwargs": kwargs, "duration": duration, "output_len": output_len})

def chat(llm, messages, trace, purpose="default", tools=None):
    if tools:
        return llm.chat_with_tools(tools, chat_history=messages[:-1], message=messages[-1])
    return llm.chat(messages)
