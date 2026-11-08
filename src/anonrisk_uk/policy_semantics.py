"""A lazy semantic tool: called with an already selected generalization decision."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .action_tools import PROMPT, provider_identity
from .detectors import ApiBudget
from .e3jsi import regex_detect
from .llm_suggestions import OpenAIReplacementSuggester
from .policy_execution import UnsupportedDecision


class LazySemanticTool:
    def __init__(self, config, client=None):
        self.config, self.client = config, client

    def __call__(self, doc, decision):
        # Credentials, cache and generation are resolved only AFTER the decision.
        identity = provider_identity(self.config)
        unit = doc.units[decision.unit]
        value = doc.text[unit.start : unit.end]
        prompt = (
            PROMPT + f"\nThe policy has ALREADY selected semantic generalization level {decision.level}. "
            "Execute this action only. Return a genuine broader category, not a synonym or translation. "
            "Omit the unit if no factual parent category can be justified."
        )
        key = hashlib.sha256(
            json.dumps([prompt, identity["model"], unit.label, value], ensure_ascii=False).encode()
        ).hexdigest()
        path = Path(self.config["out"]) / "selected_semantics" / f"{key}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            budget_path = Path(
                self.config.get("api_budget_path")
                or Path(self.config["release"]) / "artifacts/budgets/openai_policy.json"
            )
            previous = Path(self.config["release"]) / "artifacts/runs/actions_v6_openai/openai_budget.json"
            if not budget_path.exists() and previous.exists():
                budget_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(previous, budget_path)
            budget = ApiBudget(
                budget_path,
                max_usd=float(os.environ["ANONRISK_OPENAI_BUDGET_USD"]),
                max_calls=int(os.environ["ANONRISK_OPENAI_MAX_CALLS"]),
            )
            tool = OpenAIReplacementSuggester(
                budget=budget,
                allow_paid=True,
                model=identity["model"],
                prompt=prompt,
                input_usd_per_million=float(os.environ["ANONRISK_OPENAI_INPUT_USD_PER_MILLION"]),
                output_usd_per_million=float(os.environ["ANONRISK_OPENAI_OUTPUT_USD_PER_MILLION"]),
                client=self.client,
            )
            # Send only the selected entity and its type, not the entire resume.
            items = tool.suggest(
                value, [{"unit": 0, "text": value, "label": unit.label, "role": "QUASI", "start": 0, "end": len(value)}]
            )
            result = {
                "replacement": items[0].replacement if items else None,
                "model": identity["model"],
                "reason": items[0].reason if items else "No defensible parent returned",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(path)
        replacement = result["replacement"]
        if not replacement or any(s.role in {"DIRECT", "SENSITIVE"} for s in regex_detect(replacement)):
            raise UnsupportedDecision("LLM declined or introduced a direct/sensitive identifier")
        return replacement, f"openai:{result['model']}"
