"""LLM-assisted MLOps agent with a deterministic fallback router."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

from smartbet_ai.common.config import load_model_config
from smartbet_ai.common.paths import MODELS_DIR, PROJECT_ROOT


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


@dataclass
class ToolChoice:
    tool: str
    args: dict[str, Any]


class MLOpsAgent:
    """AI agent that automates basic MLOps workflows for the project."""

    def __init__(self, model_path: str | Path | None = None, openai_model: str = "gpt-4") -> None:
        config = load_model_config()
        self.mlflow_uri = config["mlops"]["mlflow_tracking_uri"]
        self.openai_model = openai_model
        self.model_path = Path(model_path or MODELS_DIR / "best_model.pt")
        self.client: OpenAI | None = None
        self.tools = {
            "check_model_performance": self._check_performance,
            "detect_data_drift": self._detect_drift,
            "trigger_retraining": self._trigger_retraining,
            "compare_model_versions": self._compare_versions,
            "generate_report": self._generate_report,
            "list_experiments": self._list_experiments,
            "check_model_status": self._check_model_status,
        }
        self.tool_descriptions = {
            "check_model_performance": "Check if the current model meets a threshold. Args: metric_name, threshold",
            "detect_data_drift": "Run drift detection on monitored features. Args: none",
            "trigger_retraining": "Trigger the local retraining script. Args: reason",
            "compare_model_versions": "Compare locally registered model versions. Args: version_a, version_b",
            "generate_report": "Generate a technical or commercial report. Args: audience",
            "list_experiments": "List MLflow experiments if MLflow is available. Args: none",
            "check_model_status": "Check whether a trained model exists and summarize metadata. Args: none",
        }

    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def _fallback_route(self, user_instruction: str) -> ToolChoice:
        lowered = user_instruction.lower()
        if "drift" in lowered:
            return ToolChoice("detect_data_drift", {})
        if "retrain" in lowered or "training" in lowered:
            return ToolChoice("trigger_retraining", {"reason": user_instruction})
        if "status" in lowered or "model file" in lowered:
            return ToolChoice("check_model_status", {})
        if "report" in lowered and "commercial" in lowered:
            return ToolChoice("generate_report", {"audience": "commercial"})
        if "report" in lowered:
            return ToolChoice("generate_report", {"audience": "technical"})
        if "compare" in lowered and "version" in lowered:
            return ToolChoice("compare_model_versions", {"version_a": 1, "version_b": 2})
        if "experiment" in lowered:
            return ToolChoice("list_experiments", {})
        return ToolChoice(
            "check_model_performance",
            {
                "metric_name": "ndcg@10",
                "threshold": load_model_config()["mlops"]["performance_threshold_ndcg"],
            },
        )

    def _llm_route(self, user_instruction: str) -> ToolChoice:
        tool_list = "\n".join([f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()])
        system_prompt = (
            "You are an MLOps automation router for a sports betting recommendation engine.\n"
            "Choose the best tool and arguments for the user's request.\n"
            f"Available tools:\n{tool_list}\n\n"
            'Respond as JSON with fields "tool" and "args".'
        )
        response = self._get_client().chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(response.choices[0].message.content)
        return ToolChoice(tool=parsed["tool"], args=parsed.get("args", {}))

    def _check_performance(self, metric_name: str = "ndcg@10", threshold: float = 0.60, **_: Any) -> dict[str, Any]:
        results = _load_json_if_exists(MODELS_DIR / "evaluation_results.json")
        if not results:
            return {"error": "No evaluation results found. Run evaluation first."}
        current_value = float(results.get(metric_name, 0.0))
        status = "PASS" if current_value >= float(threshold) else "FAIL"
        return {
            "metric": metric_name,
            "current_value": round(current_value, 4),
            "threshold": float(threshold),
            "status": status,
        }

    def _detect_drift(self, **_: Any) -> dict[str, Any]:
        from smartbet_ai.monitoring.drift import check_drift

        results, drift_detected = check_drift()
        return {
            "drift_detected": drift_detected,
            "details": results,
            "recommendation": "Retrain model immediately" if drift_detected else "No action needed",
        }

    def _trigger_retraining(self, reason: str = "manual trigger", **_: Any) -> dict[str, Any]:
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / "train.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "reason": reason,
            "return_code": result.returncode,
            "output_preview": (result.stdout or result.stderr)[-1000:],
        }

    def _compare_versions(self, version_a: int = 1, version_b: int = 2, **_: Any) -> dict[str, Any]:
        registry = _load_json_if_exists(MODELS_DIR / "model_registry.json")
        versions = registry.get("versions", [])
        indexed = {entry["version"]: entry for entry in versions}
        if version_a in indexed and version_b in indexed:
            return {
                f"version_{version_a}": indexed[version_a],
                f"version_{version_b}": indexed[version_b],
            }
        return {
            "message": "Requested versions are not both present in the local registry.",
            "available_versions": sorted(indexed.keys()),
        }

    def _generate_report(self, audience: str = "technical", **_: Any) -> dict[str, Any]:
        metrics = _load_json_if_exists(MODELS_DIR / "evaluation_results.json")
        training_summary = _load_json_if_exists(MODELS_DIR / "training_summary.json")

        if audience == "commercial":
            return {
                "audience": "commercial",
                "headline": "Recommendation engine performance summary",
                "key_findings": [
                    f"Top-10 precision: {metrics.get('precision@10', 'N/A')}",
                    f"Top-10 hit rate: {metrics.get('hit_rate@10', 'N/A')}",
                    "Synthetic-data v1 is ready for internal validation and optimisation work.",
                ],
            }

        return {
            "audience": "technical",
            "model_type": "Two-Tower Deep Learning Recommender",
            "model_path": str(self.model_path),
            "training_summary": training_summary,
            "metrics": metrics,
        }

    def _list_experiments(self, **_: Any) -> dict[str, Any]:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)
            experiments = mlflow.search_experiments()
            return {
                "experiments": [
                    {
                        "name": experiment.name,
                        "id": experiment.experiment_id,
                        "creation_time": str(experiment.creation_time),
                    }
                    for experiment in experiments
                ]
            }
        except Exception as exc:
            return {"error": f"Could not list experiments: {exc}"}

    def _check_model_status(self, **_: Any) -> dict[str, Any]:
        summary = _load_json_if_exists(MODELS_DIR / "training_summary.json")
        return {
            "model_exists": self.model_path.exists(),
            "model_path": str(self.model_path),
            "training_summary": summary,
        }

    def execute(self, user_instruction: str) -> dict[str, Any]:
        """Route a natural-language instruction to the appropriate tool."""
        try:
            tool_choice = self._llm_route(user_instruction)
        except Exception:
            tool_choice = self._fallback_route(user_instruction)

        if tool_choice.tool not in self.tools:
            return {"error": f"Unknown tool: {tool_choice.tool}"}

        result = self.tools[tool_choice.tool](**tool_choice.args)
        return {
            "tool_called": tool_choice.tool,
            "args_used": tool_choice.args,
            "result": result,
        }

    def chat(self, instruction: str) -> dict[str, Any]:
        """Print a formatted agent response for CLI use."""
        print(f"\nMLOps Agent | Instruction: '{instruction}'")
        print("-" * 60)
        result = self.execute(instruction)
        print(json.dumps(result, indent=2))
        return result


def run_agent(instruction: str, model_path: str | Path | None = None) -> dict[str, Any]:
    """Run a single agent instruction and return the result."""
    agent = MLOpsAgent(model_path=model_path)
    return agent.execute(instruction)


if __name__ == "__main__":
    agent = MLOpsAgent()
    agent.chat("Is our model performing well enough? Check if NDCG at 10 is above 0.6")
    agent.chat("Has there been any data drift recently?")
    agent.chat("Generate a report for the commercial team")
    agent.chat("What's the current status of our model?")
