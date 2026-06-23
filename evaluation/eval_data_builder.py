from chunking_service.data_to_chunks_service import data_to_chunks
from chunking_service.recursive_character_text_splitter import RecursiveChunkingStrategy
from data_loader import load_f1_data
from evaluation.eval_class import EvalCase
from pydantic import BaseModel
import json
from dataclasses import asdict

from llm_services.llm_ollama_service import LLM_Ollama_Service


class EvalCaseLLMOutput(BaseModel):
    question: str
    answer: str
    difficulty: str  # easy / medium / hard
    type: str        # definition / procedural / rule / edge-case



class EvalDatasetBuilder:

    def __init__(self, llm_service):
        self.llm = llm_service.get_llm()

    def generate_eval_cases(self, chunks, max_cases=50):

        eval_cases = []

        llm_structured = self.llm.with_structured_output(EvalCaseLLMOutput)

        for chunk in chunks[:max_cases]:

            prompt = f"""
            You are generating evaluation data for a Retrieval-Augmented Generation system.
            
            STRICT RULES:
            - Use ONLY the provided context
            - Do NOT invent facts
            - Output MUST be based only on the text
            
            TASK:
            1. Create 1 high-quality question
            2. Create exact answer from the context
            3. The question should require reading the context carefully
            
            FORMAT:
            Question: ...
            Answer: ...
            
            CONTEXT:
            {chunk.page_content}
            """

            try:
                result: EvalCaseLLMOutput = llm_structured.invoke(prompt)

                eval_cases.append(
                    EvalCase(
                        question=result.question,
                        expected_answer=result.answer,
                        expected_chunk_ids=[
                            chunk.metadata["chunk_id"]
                        ],
                    metadata={
                        "difficulty": result.difficulty,
                        "type": result.type,
                        "source_chunk_index": chunk.metadata.get("chunk_index"),
                        "source": chunk.metadata.get("source")
                    }
                    )
                )

            except Exception as e:
                print(f"Skipping chunk due to error: {e}")
                continue

        return eval_cases

    def generate_and_save(self, chunks, output_path, max_cases=50):

        eval_cases = self.generate_eval_cases(
            chunks,
            max_cases=max_cases
        )

        self.save_eval_dataset(eval_cases, file_path="eval_dataset.json")

        return eval_cases

    def save_eval_dataset(self, eval_cases, file_path: str):

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(case) for case in eval_cases],
                f,
                indent=2,
                ensure_ascii=False
            )


llm_service = LLM_Ollama_Service(model="llama3.2:1b", temperature=0)

data = load_f1_data()
chunking_strategy = RecursiveChunkingStrategy(chunk_size=1000, chunk_overlap=200)
chunks = data_to_chunks(data, chunking_strategy)


builder = EvalDatasetBuilder(llm_service)

eval_cases = builder.generate_and_save(
    chunks=chunks,
    output_path="eval_dataset.json",
    max_cases=100
)