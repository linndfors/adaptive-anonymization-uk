"""Validate completed blinded ratings and publish aggregates only, never texts/IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def summarize(review, key):
    if review.get("reviewer_type") not in {"assisted", "human"} or not review.get("reviewer"):
        raise ValueError("Reviewer identity/type must be explicit")
    rows = pd.DataFrame(review["ratings"])
    if rows.code.duplicated().any() or key.code.duplicated().any() or set(rows.code) != set(key.code):
        raise ValueError("Ratings must cover each blinded code exactly once")
    for _, row in rows.iterrows():
        if (row.professional_meaning not in {0, 1, 2, 3} or row.readability not in {0, 1, 2}
                or row.factual_distortion not in {"yes", "no", "uncertain"} or not row.notes.strip()):
            raise ValueError("Incomplete or invalid review rating")
    joined = key.merge(rows, on="code", validate="one_to_one")
    summaries = []
    for method, part in joined.groupby("method"):
        summaries.append({"method": method, "n": len(part), "persons": int(part.person.nunique()),
                          "meaning_mean_0_3": float(part.professional_meaning.mean()),
                          "meaning_counts": {str(i): int((part.professional_meaning == i).sum()) for i in range(4)},
                          "readability_mean_0_2": float(part.readability.mean()),
                          "distortion_yes": int((part.factual_distortion == "yes").sum()),
                          "distortion_no": int((part.factual_distortion == "no").sum()),
                          "distortion_uncertain": int((part.factual_distortion == "uncertain").sum())})
    return {"reviewer": review["reviewer"], "reviewer_type": review["reviewer_type"],
            "independent_human_evaluation": review["reviewer_type"] == "human",
            "people": int(key.person.nunique()), "ratings": len(rows),
            "scope": "professional field view only, not full-text grammar or privacy",
            "interpretation": "descriptive exploratory ordinal ratings; no generalization or significance claim",
            "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(json.loads(args.ratings.read_text()), pd.read_csv(args.key, dtype=str))
    report["ratings_sha256"] = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
    report["key_sha256"] = hashlib.sha256(args.key.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    pd.DataFrame(report["summary"]).drop(columns="meaning_counts").to_csv(args.out / "summary.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
