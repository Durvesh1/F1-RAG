from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel

class LLM_Base_Service(ABC):
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass