from document_loaders.document_loader import Document_Loader
from langchain_community.document_loaders import PyPDFLoader

class PDF_Document_Loader(Document_Loader):

    def load(self, path):
        return PyPDFLoader(path).load()