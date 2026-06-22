from langchain_core.prompts import ChatPromptTemplate
from llm_services.llm_service import LLM_Base_Service


class NewRagAgent:

    def __init__(self, llm_service: LLM_Base_Service):
        self.llm = llm_service.get_llm()

        self.system_prompt = """
        You are a helpful assistant that answers user questions using only the provided context.
        
        Rules:
        - Use ONLY the provided context.
        - Do NOT invent facts.
        - If context is insufficient, clearly say you don't have enough information.
        - If section IDs, page numbers, or document references exist, always include them in the answer.
        - Prefer exact matches from context when available.
        - If the context is inconsistent or conflicting, explicitly mention the conflict and do not guess.
        """

    def format_chunks(self, docs):
        """
        docs: List[RetrievedDocument]
        """
        formatted_context = []

        for i, doc in enumerate(docs):
            source = getattr(doc, "source", "unknown")
            metadata = getattr(doc, "metadata", {})

            block = f"""
                [Document {i+1}]
                Source: {source}
                Metadata: {metadata}
                
                Content:
                {doc.content}
                """
            formatted_context.append(block)

        return "\n\n".join(formatted_context)

    def get_response(self, query: str, final_docs: list):

        context = self.format_chunks(final_docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human",
             """
             Question: {query}
             Context:{context}
             """)
        ])

        chain = prompt | self.llm

        response = chain.invoke({
            "query": query,
            "context": context
        })

        return response.content