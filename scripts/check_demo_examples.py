"""Run curated synthetic Streamlit examples through the unchanged real model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anonrisk_uk.demo_highlights import highlighted_pair
from anonrisk_uk.resume_adaptive import anonymize_resume, options, render
from anonrisk_uk.resume_parser import parse_resume
from anonrisk_uk.resume_resources import DEFAULT_CONFIG, load_resume_resources
from app.demo_examples import DEMO_EXAMPLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Showcase checks already exist")
    args.out.mkdir(parents=True)
    resources = load_resume_resources(DEFAULT_CONFIG)
    results = []
    for example in DEMO_EXAMPLES:
        with threadpool_limits(limits=1):
            result = anonymize_resume(resources, example.text)
        release = result.releases["adaptive"]
        highlighted_pair(result.original, release.text, release.edits)
        doc = parse_resume(example.text, resources.codec, resources.detector)
        uniform = render(doc, {unit.group: 1 for unit in doc.units if len(options(doc, unit)) > 1})
        changed = [edit for edit in release.edits if edit["applied"]]
        value = {"key": example.key, "source_words": len(example.text.split()),
                 "missing_expected_removals": [token for token in example.direct_values if token in release.text],
                 "direct_edits": sum(edit["role"] in {"DIRECT", "SENSITIVE"} for edit in changed),
                 "adaptive_edits": sum(edit["role"] == "QUASI" for edit in changed),
                 "adaptive_steps": sum("group" in step for step in release.trace),
                 "lexical_cosine": result.diagnostics["metrics"]["lexical_retention"],
                 "no_live_pool": result.diagnostics.get("live_reference_pool_used") is False,
                 "different_from_uniform": release.text != uniform,
                 "original": example.text, "adaptive": release.text, "uniform": uniform,
                 "policy": result.diagnostics["policy"]}
        results.append(value)
        print(json.dumps({key: v for key, v in value.items() if key not in {"original", "adaptive", "uniform", "policy"}}, ensure_ascii=False), flush=True)
    model_path = ROOT / "artifacts/models/resume_action_v5/model.pkl"
    report = {"kind": "curated authored synthetic showcase, not independent model-quality evidence",
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "examples_sha256": hashlib.sha256((ROOT / "app/demo_examples.py").read_bytes()).hexdigest(),
              "examples": results}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if any(row["missing_expected_removals"] or not row["no_live_pool"] for row in results):
        raise SystemExit("One or more showcase checks failed")


if __name__ == "__main__":
    main()
