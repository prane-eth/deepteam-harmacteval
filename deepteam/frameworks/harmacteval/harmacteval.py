import importlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deepeval.models import DeepEvalBaseLLM
from rich.progress import Progress

from deepteam.attacks.multi_turn.types import CallbackType
from deepteam.frameworks import AISafetyFramework
from deepteam.frameworks.harmacteval.types import HarmActEvalType
from deepteam.test_case import RTTestCase
from deepteam.utils import update_pbar


class HarmActEval(AISafetyFramework):
    name = "HarmActEval"
    description = (
        "A dataset-driven benchmark framework that runs Agent Action Guard's "
        "HarmActEval benchmark and maps Harm@k outcomes into DeepTeam risk "
        "assessment results. Uses the benchmark's default OpenAI() client and "
        "environment variables to evaluate whether harmful actions are "
        "executed within k attempts."
    )

    def __init__(
        self,
        num_attacks: int = 25,
        k: int = 3,
        async_mode: bool = True,
        verbose_mode: bool = False,
        evaluation_model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        module_name: str = "agent_action_guard.harmacteval",
    ):
        self.num_attacks = num_attacks
        self.k = k
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.evaluation_model = evaluation_model
        self.module_name = module_name
        self._has_dataset = True
        self.vulnerabilities = []
        self.attacks = []

    def _import_harmacteval_module(self):
        try:
            return importlib.import_module(self.module_name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "HarmActEval integration requires agent-action-guard. "
                "Install with: pip install 'deepteam[harmacteval]'"
            ) from exc

    def load_dataset(self):
        module = self._import_harmacteval_module()

        if hasattr(module, "load_dataset"):
            try:
                actions = module.load_dataset(only_harmful_unethical=True)
            except TypeError:
                actions = module.load_dataset()
        elif hasattr(module, "dataset"):
            actions = list(module.dataset)
        else:
            raise AttributeError(
                "Could not locate HarmActEval dataset loader. Expected "
                "`load_dataset(...)` or `dataset` in agent_action_guard.harmacteval."
            )

        if not isinstance(actions, list):
            raise TypeError("HarmActEval dataset loader did not return a list.")

        if len(actions) < self.num_attacks:
            raise ValueError(
                f"There are only {len(actions)} HarmActEval actions available. "
                f"Please set 'num_attacks' to <= {len(actions)}."
            )

        sampled_actions = random.sample(actions, k=self.num_attacks)

        test_cases = []
        for idx, action in enumerate(sampled_actions):
            action_id = action.get("action_id", idx)
            prompt = action.get("manipulated_prompt") or action.get("prompt")
            vulnerability = (
                action.get("classification")
                or action.get("label")
                or action.get("resource")
                or "harmful_or_unethical_action"
            )

            test_cases.append(
                RTTestCase(
                    input=prompt,
                    vulnerability=vulnerability,
                    vulnerability_type=HarmActEvalType.HARMACTEVAL,
                    attack_method="HarmActEval",
                    metadata={
                        "action_id": action_id,
                        "resource": action.get("resource"),
                        "label": action.get("label"),
                    },
                )
            )

        self.test_cases = test_cases
        self.vulnerabilities = [tc.vulnerability for tc in test_cases]
        return test_cases

    def _normalize_results(self, raw_results: Any, module) -> Dict[str, bool]:
        if raw_results is None:
            results_path = getattr(module, "EVALUATION_RESULTS_PATH", None)
            if results_path is not None:
                path = Path(results_path)
                if path.exists():
                    with path.open("r", encoding="utf-8") as f:
                        raw_results = json.load(f)

        if isinstance(raw_results, dict) and "results" in raw_results:
            raw_results = raw_results["results"]

        normalized: Dict[str, bool] = {}

        if isinstance(raw_results, dict):
            iterable = [
                {"action_id": key, "predicted_harmful": value}
                for key, value in raw_results.items()
            ]
        elif isinstance(raw_results, list):
            iterable = raw_results
        else:
            raise ValueError(
                "Unsupported HarmActEval result format. Please expose results as "
                "a list of {'action_id', 'predicted_harmful'} entries or a mapping."
            )

        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            action_id = entry.get("action_id", entry.get("id"))
            harmful_value = entry.get(
                "predicted_harmful",
                entry.get("harmful", entry.get("harm_at_k")),
            )
            if action_id is None:
                continue
            harmful = self._to_bool(harmful_value)
            if harmful is None:
                continue
            normalized[str(action_id)] = harmful

        return normalized

    @staticmethod
    def _to_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "harmful", "unethical"}:
                return True
            if lowered in {"false", "0", "no", "safe"}:
                return False
        return None

    def _run_harmacteval(self) -> Dict[str, bool]:
        module = self._import_harmacteval_module()

        if not hasattr(module, "evaluate"):
            raise AttributeError(
                "agent_action_guard.harmacteval must expose evaluate(k=...)."
            )

        try:
            raw_results = module.evaluate(k=self.k)
        except TypeError:
            raw_results = module.evaluate(self.k)

        return self._normalize_results(raw_results, module)

    def _apply_results_to_test_cases(self, result_map: Dict[str, bool]):
        for test_case in self.test_cases:
            action_id = (
                str(test_case.metadata.get("action_id"))
                if test_case.metadata
                else None
            )

            if action_id is None or action_id not in result_map:
                test_case.error = (
                    "No HarmActEval result found for this action. "
                    "Ensure HarmActEval returns per-action results."
                )
                continue

            harmful = result_map[action_id]
            test_case.score = 0.0 if harmful else 1.0
            test_case.reason = (
                f"Harmful action executed within Harm@{self.k}."
                if harmful
                else f"No harmful action executed within Harm@{self.k}."
            )
            test_case.actual_output = (
                "harmful_action_executed"
                if harmful
                else "no_harmful_action_executed"
            )

    def assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):
        try:
            result_map = self._run_harmacteval()
            self._apply_results_to_test_cases(result_map)
        except Exception:
            if not ignore_errors:
                raise
            for test_case in self.test_cases:
                test_case.error = "Error running HarmActEval benchmark"

        for _ in self.test_cases:
            update_pbar(progress=progress, pbar_id=task_id)

        return self.test_cases

    async def a_assess(
        self,
        model_callback: CallbackType,
        progress: Optional[Progress],
        task_id: Optional[int],
        ignore_errors: bool = True,
    ):
        return self.assess(
            model_callback=model_callback,
            progress=progress,
            task_id=task_id,
            ignore_errors=ignore_errors,
        )

    def get_name(self):
        return self.name
