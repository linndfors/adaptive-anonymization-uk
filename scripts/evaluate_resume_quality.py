"""Paired, synthetic extraction/utility diagnostics. Not an S2 privacy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from anonrisk_uk.demo import load_demo_resources
from anonrisk_uk.entity_demo import anonymize_entity_text
from anonrisk_uk.resume_adaptive import CORE, VERSION, compare_resume
from anonrisk_uk.resume_parser import FIELDS, extractor_from_env

ROOT = Path(__file__).resolve().parents[1]


class RecordingExtractor:
    """Save accepted extraction for reproducible synthetic-fixture rerendering only."""

    def __init__(self, extractor, path):
        self.extractor, self.path, self.model = extractor, path, extractor.model
        self.rejected_entities = 0

    def extract(self, text):
        data = self.extractor.extract(text)
        self.rejected_entities = self.extractor.rejected_entities
        self.path.write_text(json.dumps({"model": self.model, "data": data,
                                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                        "rejected_entities": self.rejected_entities}, ensure_ascii=False, indent=2))
        return data


class ReplayExtractor:
    """Rerun local parsing/search against a recorded synthetic API response, no API call."""

    def __init__(self, path):
        self.record = json.loads(path.read_text())
        self.model = self.record["model"]
        self.rejected_entities = self.record["rejected_entities"]

    def extract(self, text):
        if hashlib.sha256(text.encode()).hexdigest() != self.record["text_sha256"]:
            raise ValueError("Cached extraction belongs to a different fixture text")
        return self.record["data"]


def decode(annotated):
    text, entities, cursor = "", [], 0
    for match in re.finditer(r"\[\[([A-Z_]+)\|(.+?)\]\]", annotated, re.DOTALL):
        text += annotated[cursor:match.start()]
        start = len(text)
        text += match[2]
        entities.append({"start": start, "end": len(text), "label": match[1], "text": match[2]})
        cursor = match.end()
    return text + annotated[cursor:], entities


def measure(gold, release):
    direct = [g for g in gold if FIELDS[g["label"]][0] in {"DIRECT", "SENSITIVE"}]
    protected = [g for g in gold if FIELDS[g["label"]][1] in CORE]
    # Formatting whitespace between two masked chunks is not a leaked identifier.
    covered = {e["start"] + i for e in release.edits if e["applied"] and e["role"] in {"DIRECT", "SENSITIVE"}
               for i, char in enumerate(e["text"]) if not char.isspace()}
    expected = {e["start"] + i for e in direct for i, char in enumerate(e["text"]) if not char.isspace()}
    exact = sum(any(e["label"] == g["label"] and e["start"] == g["start"] and e["end"] == g["end"]
                    for e in release.edits) for g in gold)
    leaked = sum(not {g["start"] + i for i, char in enumerate(g["text"]) if not char.isspace()} <= covered
                 for g in direct)
    return {
        "direct_entities": len(direct), "leaked_direct_entities": leaked,
        "direct_chars": len(expected), "masked_direct_chars": len(expected & covered),
        "masked_chars": len(covered), "gold_entities": len(gold), "exact_typed_entities": exact,
        "core_facts": len(protected), "preserved_core_facts": sum(g["text"] in release.text for g in protected),
        "lexical_similarity": release.retention,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research_entity_policy_v4c_local.json")
    parser.add_argument("--fixtures", default="tests/fixtures/resume_quality.json")
    parser.add_argument("--split", choices=["dev", "challenge", "all"], default="dev")
    parser.add_argument("--openai", action="store_true", help="Send synthetic fixture texts to configured OpenAI model")
    parser.add_argument("--replay-extractions", type=Path, help="Recorded synthetic extractions directory; no API calls")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.openai and args.replay_extractions:
        parser.error("Choose live OpenAI or replay, not both")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = Path(args.fixtures).read_bytes()
    cases = [c for c in json.loads(source) if args.split == "all" or c["split"] == args.split]
    extractor = extractor_from_env(ROOT) if args.openai else None
    model_name = extractor.model if extractor else None
    if extractor:
        (out / "extractions").mkdir()
    rows, outputs = [], []
    with threadpool_limits(limits=1):
        resources = load_demo_resources(args.config)
        for case in cases:
            text, gold = decode(case["annotated"])
            legacy = anonymize_entity_text(resources, text, mode="legacy")
            local = compare_resume(resources, text)
            results = {"legacy_v4c": legacy.releases["adaptive_model"],
                       "guarded_local": local.releases["adaptive_guarded"],
                       "local_direct_only": local.releases["direct_only"],
                       "local_quasi_generalized": local.releases["quasi_generalized"]}
            diagnostics = {"local": local.diagnostics}
            if extractor or args.replay_extractions:
                active = (ReplayExtractor(args.replay_extractions / f"{case['id']}.json") if args.replay_extractions
                          else RecordingExtractor(extractor, out / "extractions" / f"{case['id']}.json"))
                if model_name and model_name != active.model:
                    raise ValueError("Replay cannot mix different models")
                model_name = active.model
                api = compare_resume(resources, text, extractor=active)
                results.update({"guarded_api": api.releases["adaptive_guarded"],
                                "api_direct_only": api.releases["direct_only"],
                                "api_quasi_generalized": api.releases["quasi_generalized"]})
                diagnostics["api"] = api.diagnostics
            for name, release in results.items():
                rows.append({"id": case["id"], "method": name, **measure(gold, release)})
            outputs.append({"id": case["id"], "diagnostics": diagnostics,
                            "releases": {k: {"text": v.text, "edits": v.edits, "trace": v.trace}
                                         for k, v in results.items()}})
            # Save completed cases: an API error never loses already-paid results.
            pd.DataFrame(rows).to_csv(out / "per_resume.csv", index=False)
            (out / "outputs.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
            print(case["id"], "done", flush=True)
    frame = pd.DataFrame(rows)
    summary = []
    for method, part in frame.groupby("method"):
        totals = part.sum(numeric_only=True)
        summary.append({"method": method, "n": len(part),
                        "direct_char_recall": totals.masked_direct_chars / max(totals.direct_chars, 1),
                        "direct_mask_precision": totals.masked_direct_chars / max(totals.masked_chars, 1),
                        "direct_entity_leak_rate": totals.leaked_direct_entities / max(totals.direct_entities, 1),
                        "exact_type_recall": totals.exact_typed_entities / max(totals.gold_entities, 1),
                        "core_verbatim_survival": totals.preserved_core_facts / max(totals.core_facts, 1),
                        "lexical_similarity": part.lexical_similarity.mean()})
    report = {"method_version": VERSION, "fixture_sha256": hashlib.sha256(source).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                  ROOT / "src/anonrisk_uk/resume_parser.py", ROOT / "src/anonrisk_uk/resume_adaptive.py"]},
              "split": args.split, "model": model_name,
              "extraction_replay": str(args.replay_extractions) if args.replay_extractions else None,
              "direct_coverage_definition": "non-whitespace source characters; partial entity coverage is a leak",
              "protocol": "synthetic diagnostic; NOT independent WorkUA S2 or evidence of superiority over L2",
              "summary": summary}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
