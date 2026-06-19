from langchain_ollama import ChatOllama

from llm_service import LLM_Base_Service


class LLM_Ollama_Service(LLM_Base_Service):

    def run(self, prompt):
        return ChatOllama(model="llama3.2:1b", temperature=0)
