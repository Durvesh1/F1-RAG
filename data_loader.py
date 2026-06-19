import os
from pathlib import Path

from document_loaders.document_docx_loader import DOCX_Document_Loader
from document_loaders.document_pdf_loader import PDF_Document_Loader
from document_loaders.document_txt_loader import TEXT_Document_Loader

path = "./data"
def load_data(path=path):
    path = "./data"
    all_content = ""
    count = 0
    max = 2

    for filename in os.listdir(path):
        if count <= max:
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) and filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    all_content +=f.read()+"\n\n"
                    count+=1
    return all_content

def load_f1_data():
    file_path = "./data/f1-regulations.pdf"
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return PDF_Document_Loader().load(file_path)
    elif ext == ".doc":
        return DOCX_Document_Loader().load(file_path)
    elif ext == ".txt":
        return TEXT_Document_Loader().load(file_path)
    else:
        raise ValueError("Unsupported file type")

    # loader = PyPDFLoader("./data/f1-regulations.pdf")
    # doc = loader.load()

    # return doc


