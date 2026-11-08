"""Anonymize a UTF-8 resume with the same engine as the Streamlit demo."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from threadpoolctl import threadpool_limits

from anonrisk_uk.resume_adaptive import anonymize_resume
from anonrisk_uk.resume_parser import extractor_from_env
from anonrisk_uk.resume_resources import load_resume_resources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Optional JSON audit containing ORIGINAL sensitive fragments")
    parser.add_argument("--config", default="configs/adaptive_resume.json")
    parser.add_argument("--openai", action="store_true", help="Send the input to the configured OpenAI extractor")
    args = parser.parse_args()
    for path in (args.output, args.audit):
        if path and path.exists():
            parser.error(f"Output already exists: {path}")
    text = args.input.read_text(encoding="utf-8")
    with threadpool_limits(limits=1):
        resources = load_resume_resources(args.config)
        result = anonymize_resume(resources, text, extractor=extractor_from_env(ROOT) if args.openai else None)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(result.releases["adaptive"].text)
    if args.audit:
        with args.audit.open("x", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, ensure_ascii=False, indent=2)
    print(json.dumps(result.diagnostics, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
