from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_community.retrievers import BM25Retriever

from llm_services.llm_service import LLM_Base_Service


class RelevanceGrade(BaseModel):
    relevance: str

class RetrieverAgent:

    def __init__(self, vector_store, llm_service: LLM_Base_Service, raw_data):
        self.vector_store = vector_store

        self.llm = llm_service.get_llm()

        self.raw_data = raw_data


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

    def bm25_retrieve(self):
        bm25_retriever = BM25Retriever.from_documents(self.raw_data)
        bm25_retriever.k = 3

        results = bm25_retriever.invoke("How many team members are allowed in the signalling area?")
        res = []
        for result in results:
            dict = defaultdict()
            dict["content"] = result.page_content
            res.append(dict)
        return res

    def get_chunks(self, query:str , top_entries = 3,rerank = True):
        chunks = self.vector_store.similarity_search_with_score(query, k = top_entries * (2 if rerank else 1))

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

        temp = revised_chunks[:top_entries]

        bm25_chunks = self.bm25_retrieve()

        return bm25_chunks+temp









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