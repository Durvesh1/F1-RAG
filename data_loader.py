from pathlib import Path

from document_loaders.document_docx_loader import DOCX_Document_Loader
from document_loaders.document_pdf_loader import PDF_Document_Loader
from document_loaders.document_txt_loader import TEXT_Document_Loader


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


