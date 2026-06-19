from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from llm_ollama_service import LLM_Ollama_Service
from llm_service import LLM_Base_Service


class RagAgent:
    def __init__(self, llm_service:LLM_Base_Service):
        self.llm = llm_service.get_llm()

        self.system_prompt = """
        You are a helpful assistant and would help the user by assessing the context given to you for a user question
        and give response back to the user question based on the data that is present in the context.
        Your response should only be based on the context that is shared with you and you should not invent any
        facts that are not part of the document. 
        Also if there is any section ID, number or relevant document number information present in the context, you should
        quote that relevant information as well.
        
        If the user question and the context that provided to you does not make sense or is inconsistent then
        you should say you do not have all the information for the user query and do not give any further response. 
        
        """

    def format_chunks(self, data_chunks):
        chunks = data_chunks
        context = ""

        for chunk in chunks:
            context = context + "\n" + chunk["content"]

        return context

    def get_response(self, query, top_chunks):

        query = query
        context = self.format_chunks(top_chunks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """
                    Query: {query}
                    Context: {context}
                    """)
        ]
        )

        chain = prompt | self.llm

        response = chain.invoke({"query": query, "context": context})

        return response.content

