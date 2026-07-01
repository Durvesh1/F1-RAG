from abc import ABC, abstractmethod


class BaseGuardrail(ABC):

    @abstractmethod
    def run(self, data):
        pass