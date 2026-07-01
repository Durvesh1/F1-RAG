import re

from guardrails.base_guardrail import BaseGuardrail


class QueryValidator(BaseGuardrail):

    MIN_LENGTH = 3
    MAX_LENGTH = 1000

    def run(self, query: str):

        query = query.strip()

        if len(query) < self.MIN_LENGTH:
            raise ValueError("Query too short.")

        if len(query) > self.MAX_LENGTH:
            raise ValueError("Query too long.")

        if re.fullmatch(r"[0-9]+", query):
            raise ValueError("Invalid query.")

        if re.fullmatch(r"[^\w]+", query):
            raise ValueError("Invalid query.")

        return query