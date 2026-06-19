from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

class RelevanceGrade(BaseModel):
    relevance: str

class RetrieverAgent:

    def __init__(self, vector_store, model="llama3.2:1b", temperature=0):
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature

        self.llm = ChatOllama(model=model, temperature=temperature)


        self.system_prompt = """
        You are a grader assessing relevance of a retrieved document to a user question.
         If the document contains keyword(s) or semantic meaning related to the user question,
         grade it as relevant.
         It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """


    def get_relevance_score(self, query:str, chunks: str) -> str:

        structured_llm = self.llm.with_structured_output(
            RelevanceGrade
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
            Query: {query}
            Chunks: {chunks}
            """)
        ]
        )

        chain = prompt | structured_llm

        response = chain.invoke({"query": query, "chunks": chunks})

        return response.relevance

    def get_chunks(self, query:str , top_entries = 3,rerank = True):
        chunks = self.vector_store.similarity_search_with_score(query, k = top_entries * (2 if rerank else 1))
        # dict = defaultdict()

        revised_chunks = []

        for doc, score in chunks:
            dict = defaultdict()
            dict["content"] = doc.page_content

            if rerank:
                grade = self.get_relevance_score(query, doc.page_content)
                if grade == "yes":
                    dict["score"] = 1-score
                    revised_chunks.append(dict)
            else:
                dict["score"] = 1 - score
                revised_chunks.append(dict)

        revised_chunks.sort(reverse = True, key = lambda x: x["score"])

        return revised_chunks[:top_entries]

def get_retriever_agent(vector_store, model="llama3.2:1b", temperature=0):
    return RetrieverAgent(vector_store, model, temperature)










# vector_store = vectorize_data("")
#
# retriever = RetrieverAgent(vector_store)
#
# chunks = vector_store.as_retriever(kwargs={"k":3})
# query = "Aircraft operator report"
# retrieved_chunks = chunks.invoke(query)
#
# context = "\n\n".join(
#     doc.page_content
#     for doc in retrieved_chunks
# )
#
# retriever.get_relevance_score("",context)