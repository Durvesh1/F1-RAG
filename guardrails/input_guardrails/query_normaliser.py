import re

from guardrails.base_guardrail import BaseGuardrail


class QueryNormalizer(BaseGuardrail):

    def run(self, query: str):

        query = query.strip()

        query = re.sub(r"\s+", " ", query)

        return query