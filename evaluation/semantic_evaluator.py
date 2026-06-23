from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticAnswerEvaluator:

    def __init__(self,embedding_model):
        self.embedding_model = embedding_model

    def evaluate(self,generated_answer,expected_answer):

        generated_embedding = (
            self.embedding_model
            .embed_query(
                generated_answer
            )
        )

        expected_embedding = (
            self.embedding_model
            .embed_query(
                expected_answer
            )
        )

        similarity = cosine_similarity(

            np.array(
                generated_embedding
            ).reshape(1, -1),

            np.array(
                expected_embedding
            ).reshape(1, -1)

        )[0][0]

        return float(similarity)