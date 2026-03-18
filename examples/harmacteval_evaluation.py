from deepteam import red_team
from deepteam.frameworks import HarmActEval
from deepeval.models import AzureOpenAIModel, GPTModel


model = GPTModel(temperature=1)

framework = HarmActEval(k=1)


if __name__ == "__main__":
    risk_assessment = red_team(
        model_callback=model,  # type: ignore
        framework=framework,
        async_mode=False,
        ignore_errors=False,
    )

    total = len(risk_assessment.test_cases)
    safe = sum(1 for tc in risk_assessment.test_cases if tc.score == 1.0)

    print(
        f"HarmActEval complete: {safe}/{total} actions remained safe within Safe@{framework.k}."
    )
