from langchain_ollama import ChatOllama

from llm_services.llm_service import LLM_Base_Service


class LLM_Ollama_Service(LLM_Base_Service):

    def __init__(self, model, temperature):
        self.llm = ChatOllama(model=model, temperature=temperature)

    def get_llm(self):
        return self.llm
