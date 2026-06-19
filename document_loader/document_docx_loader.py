from document_loader import Document_Loader
from langchain_community.document_loaders import Docx2txtLoader


class DOCX_Document_Loader(Document_Loader):

    def load(self, path):
        return Docx2txtLoader(path).load()