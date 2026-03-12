import types

from deepteam.frameworks import HarmActEval
from deepteam import red_team


class TestHarmActEval:
    def test_harmacteval_init(self):
        framework = HarmActEval()
        assert framework is not None
        assert framework.name == "HarmActEval"

    def test_harmacteval_load_dataset_and_assess(self, monkeypatch):
        fake_module = types.SimpleNamespace(
            load_dataset=lambda only_harmful_unethical=True: [
                {
                    "action_id": 1,
                    "prompt": "do thing",
                    "manipulated_prompt": "do harmful thing",
                    "classification": "harmful",
                    "label": "harmful_tool",
                    "resource": "filesystem",
                },
                {
                    "action_id": 2,
                    "prompt": "do other thing",
                    "manipulated_prompt": "do unsafe thing",
                    "classification": "unethical",
                    "label": "unsafe_tool",
                    "resource": "network",
                },
            ],
            evaluate=lambda k=3: [
                {"action_id": 1, "predicted_harmful": True},
                {"action_id": 2, "predicted_harmful": False},
            ],
        )

        monkeypatch.setattr(
            "importlib.import_module", lambda _: fake_module
        )

        framework = HarmActEval(num_attacks=2)
        framework.load_dataset()

        assert len(framework.test_cases) == 2

        def unused_model_callback(prompt: str) -> str:
            return "ignored"

        test_cases = framework.assess(
            model_callback=unused_model_callback,
            progress=None,
            task_id=None,
            ignore_errors=False,
        )

        assert test_cases[0].score == 0.0
        assert test_cases[1].score == 1.0
        assert test_cases[0].error is None
        assert test_cases[1].error is None

    def test_harmacteval_red_team_integration(self, monkeypatch):
        fake_module = types.SimpleNamespace(
            load_dataset=lambda only_harmful_unethical=True: [
                {
                    "action_id": 10,
                    "manipulated_prompt": "trigger tool",
                    "classification": "harmful",
                    "label": "dangerous_tool",
                    "resource": "tool_api",
                }
            ],
            evaluate=lambda k=3: [{"action_id": 10, "predicted_harmful": True}],
        )

        monkeypatch.setattr(
            "importlib.import_module", lambda _: fake_module
        )

        def model_callback(prompt: str) -> str:
            return "unused"

        risk_assessment = red_team(
            model_callback=model_callback,
            framework=HarmActEval(num_attacks=1),
            async_mode=False,
            ignore_errors=False,
        )

        assert risk_assessment is not None
        assert len(risk_assessment.test_cases) == 1
        assert risk_assessment.test_cases[0].score == 0.0
