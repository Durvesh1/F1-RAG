import re

from guardrails.base_guardrail import BaseGuardrail


class PromptInjectionGuard(BaseGuardrail):

    PATTERNS = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"system prompt",
        r"developer message",
        r"reveal your prompt",
        r"bypass",
        r"jailbreak"
    ]

    def run(self, query):

        lower = query.lower()

        for pattern in self.PATTERNS:

            if re.search(pattern, lower):

                raise ValueError(
                    "Potential prompt injection detected."
                )

        return query