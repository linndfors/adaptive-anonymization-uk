"""Recover the completed CPU comparisons after the optional-category pairing bug.

Only the exact pairing-only source change is accepted. Original input/code hashes
must match the failed runner state. Missing expensive checkpoints abort recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_LINE = '                *(["adaptive_category"] if method == "adaptive_model" else []),'
NEW_LINE = (
    '                *(["adaptive_category"] if method == "adaptive_model" and '
    '"category" in config.get("strategies", ["category", "model"]) else []),'
)


def verify_source(previous, current):
    if previous.count(OLD_LINE) != 1 or previous.replace(OLD_LINE, NEW_LINE) != current:
        raise ValueError("Source differs beyond the supported pairing-only fix; recovery aborted")


def previous_signature(config_path, config, old_source):
    # Match run_cpu_pipeline.signature exactly, substituting only the verified
    # pre-fix research.py bytes. Never ignore changes to any other input or code.
    digest = hashlib.sha256()
    paths = [config_path, PROJECT_ROOT / "uv.lock", PROJECT_ROOT / "pyproject.toml",
             PROJECT_ROOT / "scripts/run_cpu_pipeline.py"]
    paths.extend(sorted((PROJECT_ROOT / "src/anonrisk_uk").glob("*.py")))
    for key in ("workua_path", "e3_path", "vacancies_path"):
        if config.get(key):
            paths.append(Path(config[key]))
    for path in paths:
        digest.update(str(path).encode())
        if path == PROJECT_ROOT / "src/anonrisk_uk/research.py":
            digest.update(old_source)
        elif path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            digest.update(b"missing")
    return digest.hexdigest()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research_cpu.json")
    parser.add_argument("--run", action="store_true", help="Run recovery after validation; otherwise only inspect")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from anonrisk_uk import research

    config_path = args.config.resolve()
    config = research.resolve_config(config_path)
    config["allow_paid_api"] = False
    out = Path(config["out"])
    state_path = out / "cpu_pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    old_source = (out / "recovery/research_before_pairing_fix.py.txt").read_bytes()
    source_path = Path(research.__file__).resolve()
    verify_source(old_source.decode("utf-8"), source_path.read_text(encoding="utf-8"))
    if previous_signature(config_path, config, old_source) != state["signature"]:
        raise ValueError("Original runner signature no longer matches inputs/code; recovery aborted")
    if state["stages"]["pipeline"]["status"] == "running":
        raise ValueError("Pipeline is marked running; recovery must not overlap another run")
    if not config.get("resume_checkpoints"):
        raise ValueError("Recovery requires resume_checkpoints=true")
    print("Verified: only the optional-category comparison changed; original inputs/code match.", flush=True)
    for stage in ("validation", "test_release", "utility_documents", "utility_release"):
        for directory in sorted((out / "checkpoints").glob("*/" + stage)):
            print(f"{directory}: {sum(1 for _ in directory.glob('*.pkl'))} saved results", flush=True)
    if not args.run:
        print("Inspection only. Add --run to recover; expensive missing checkpoints will cause an error.")
        return 0

    spec = importlib.util.spec_from_file_location("cpu_runner", PROJECT_ROOT / "scripts/run_cpu_pipeline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    fingerprint = research._fingerprint
    checkpoint = research._checkpoint

    def compatible_fingerprint(path):
        if Path(path).resolve() == source_path:
            return hashlib.sha256(old_source).hexdigest()
        return fingerprint(path)

    def cached_only(config, stage, key, compute):
        def missing():
            raise RuntimeError(f"Missing compatible checkpoint {stage}/{key}; refusing expensive recomputation")
        return checkpoint(config, stage, key, missing)

    research._fingerprint = compatible_fingerprint
    research._checkpoint = cached_only
    previous_attempt = dict(state["stages"]["pipeline"])
    record = {"status": "running", "started_at": datetime.now(UTC).isoformat(),
              "previous_attempt": previous_attempt, "recovery": "optional-category-pairing-only"}
    state["stages"]["pipeline"] = record
    runner.save_state(state_path, state)
    started = time.monotonic()
    log_path = out / "logs/pipeline_pairing_recovery.log"
    print(f"Recovery log: {log_path}", flush=True)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== Recovery {record['started_at']} ===\n")
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                research.run_pipeline(config)
        record.update(status="completed", exit_code=0)
        # The narrow compatibility check also covers the first five stages.
        state["signature"] = runner.signature(config_path, config)
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        research._fingerprint = fingerprint
        research._checkpoint = checkpoint
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        runner.save_state(state_path, state)
    print(f"Pipeline recovery completed. Results: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
