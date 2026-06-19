import os

from langchain_community.document_loaders import PyPDFLoader

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

    loader = PyPDFLoader("./data/f1-regulations.pdf")
    doc = loader.load()

    return doc


