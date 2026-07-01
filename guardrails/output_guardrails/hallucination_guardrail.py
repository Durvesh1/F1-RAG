from guardrails.base_guardrail import BaseGuardrail


class HallucinationGuard(BaseGuardrail):

    def run(self,docs):
        if not docs:
            raise ValueError(
                "No supporting context."
            )
        else:
            pass