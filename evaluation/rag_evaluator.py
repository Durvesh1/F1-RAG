class RagEvaluator:

    def __init__(self,retriever_agent,rag_agent,answer_evaluator):

        self.retriever_agent = (
            retriever_agent
        )

        self.rag_agent = rag_agent

        self.answer_evaluator = (
            answer_evaluator
        )

    def evaluate_case(
        self,
        case
    ):

        docs = (
            self.retriever_agent.retrieve(
                case.question
            )
        )

        answer = (
            self.rag_agent.get_response(
                case.question,
                docs
            )
        )

        score = (
            self.answer_evaluator
            .evaluate(
                answer,
                case.expected_answer
            )
        )

        return {

            "question":
                case.question,

            "answer_score":
                score,

            "generated_answer":
                answer
        }