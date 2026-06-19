from abc import ABC, abstractmethod

class Document_Loader(ABC):

    @abstractmethod
    def load(self, path):
        pass