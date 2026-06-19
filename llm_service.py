from abc import ABC, abstractmethod

class LLM_Base_Service(ABC):
    @abstractmethod
    def run(self, prompt):
        pass