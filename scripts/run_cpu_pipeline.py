"""Run all six CPU research stages with logs and resumable stage state."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    "detection": ["detection/manifest.json", "detection/selected_detector.json", "detection/metrics.csv"],
    "workua": ["workua/bundle.pkl", "workua/manifest.json", "workua/split.csv"],
    "risk-labels": [
        "pipeline/risk_label_manifest.json", "pipeline/unit_training_features.csv",
        "pipeline/unit_training_labels.csv", "pipeline/unit_validation_features.csv",
        "pipeline/unit_validation_labels.csv",
    ],
    "risk-train": [
        "pipeline/unit_risk_model.pkl", "pipeline/risk_model_selection.json",
        "pipeline/risk_model_validation.csv", "pipeline/risk_learning_curve.csv",
    ],
    "vacancy-train": [],
    "pipeline": [
        "pipeline/manifest.json", "pipeline/operating_point.json", "pipeline/validation_tradeoffs.csv",
        "pipeline/test_summary.csv", "pipeline/test_intervals.csv", "pipeline/vacancy_summary.csv",
    ],
}


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def signature(config_path, config):
    """Invalidate stage completion when configuration, code, or input files change."""
    digest = hashlib.sha256()
    paths = [config_path, PROJECT_ROOT / "uv.lock", PROJECT_ROOT / "pyproject.toml", Path(__file__)]
    paths.extend(sorted((PROJECT_ROOT / "src/anonrisk_uk").glob("*.py")))
    for key in ("workua_path", "e3_path", "vacancies_path"):
        if config.get(key):
            paths.append(Path(config[key]))
    for path in paths:
        digest.update(str(path).encode())
        if path.is_file():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            digest.update(b"missing")
    if config.get("llm_provider") == "openai":
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from anonrisk_uk.action_tools import provider_identity
        digest.update(json.dumps(provider_identity(config), sort_keys=True).encode())
    return digest.hexdigest()


def run_stage(stage, config_path, log_path, interval, timeout=None, allow_paid=False):
    started = time.monotonic()
    stop = threading.Event()
    expired = threading.Event()

    def heartbeat():
        next_notice = interval
        while not stop.wait(min(interval, 1)):
            elapsed = time.monotonic() - started
            if timeout and elapsed > timeout:
                expired.set()
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return
            if elapsed >= next_notice:
                print(f"\n[{stage}] працює; минуло {elapsed / 60:.1f} хв", flush=True)
                next_notice += interval

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now(UTC).isoformat()} {stage} ===\n")
        log.flush()
        command = [sys.executable, "-u", "-m", "anonrisk_uk.cli", stage, "--config", str(config_path)]
        if allow_paid:
            command.append("--allow-paid")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
        try:
            assert process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            # Read chunks, not lines: tqdm uses carriage returns for live progress.
            while chunk := process.stdout.read1(4096):
                text = decoder.decode(chunk)
                print(text, end="", flush=True)
                log.write(text)
                log.flush()
            tail = decoder.decode(b"", final=True)
            print(tail, end="", flush=True)
            log.write(tail)
            code = process.wait()
            if expired.is_set():
                raise RuntimeError(f"Stage exceeded the configured time limit ({timeout}s); checkpoints retained")
            return code
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            stop.set()
            worker.join()
            if process.stdout is not None:
                process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/research_cpu.json")
    parser.add_argument("--restart", action="store_true", help="Rerun every stage; retain item-level checkpoints")
    parser.add_argument("--allow-paid", action="store_true", help="Enable explicitly configured, budgeted API tools")
    parser.add_argument("--progress-seconds", type=float, default=30, help="Heartbeat interval (default: 30)")
    args = parser.parse_args()
    if args.progress_seconds <= 0:
        parser.error("--progress-seconds must be positive")

    # Keep --help usable without loading the scientific stack.
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from anonrisk_uk.research import resolve_config

    config_path = args.config.resolve()
    config = resolve_config(config_path)
    out = Path(config["out"])
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = out / "cpu_pipeline_state.json"
    print(f"Результати: {out}\nЛоги: {logs}\nПеревіряю конфігурацію, код і вхідні дані…", flush=True)
    current_signature = signature(config_path, config)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if args.restart or state.get("signature") != current_signature:
        state = {"signature": current_signature, "config": str(config_path), "stages": {}}
    save_state(state_path, state)

    # vacancy_model_path is interpreted relative to the repository by the CLI.
    model = Path(config["vacancy_model_path"])
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    outputs = {stage: [out / name for name in names] for stage, names in OUTPUTS.items()}
    outputs["vacancy-train"] = [model, model.with_suffix(".manifest.json")]
    if not config.get("vacancies_path"):
        outputs["pipeline"].remove(out / "pipeline/vacancy_summary.csv")

    invalidated = False
    for index, stage in enumerate(OUTPUTS, 1):
        if stage == "vacancy-train" and config.get("skip_vacancy_training", False):
            state["stages"][stage] = {"status": "skipped", "reason": "vacancies disabled in bounded profile"}
            save_state(state_path, state)
            print(f"[{index}/6] {stage}: пропущено за конфігурацією.", flush=True)
            continue
        record = state["stages"].get(stage, {})
        ready = all(path.is_file() and path.stat().st_size for path in outputs[stage])
        if not invalidated and record.get("status") == "completed" and ready:
            print(f"[{index}/6] {stage}: уже завершено, пропускаю.", flush=True)
            continue
        # Once an upstream stage is rerun, all downstream stages must be rerun too.
        invalidated = True
        for downstream in list(OUTPUTS)[index - 1:]:
            state["stages"].pop(downstream, None)
        record = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
        state["stages"][stage] = record
        save_state(state_path, state)
        log_path = logs / f"{stage}.log"
        print(f"\n[{index}/6] Запускаю {stage}. Лог: {log_path}", flush=True)
        started = time.monotonic()
        try:
            if config.get("max_stage_seconds") or args.allow_paid:
                code = run_stage(stage, config_path, log_path, args.progress_seconds,
                                 config.get("max_stage_seconds"), args.allow_paid)
            else:
                code = run_stage(stage, config_path, log_path, args.progress_seconds)
            record["exit_code"] = code
            if code:
                raise RuntimeError(f"Код виходу: {code}")
            missing = [str(path) for path in outputs[stage] if not path.is_file() or not path.stat().st_size]
            if missing:
                raise RuntimeError(f"Відсутні результати: {missing}")
            record["status"] = "completed"
        except (OSError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
            record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            record["error"] = str(exc)
            print(f"\n{stage}: зупинено. {exc}\nЛог: {log_path}\nДля продовження повторіть ту саму команду.", flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
            save_state(state_path, state)
        print(f"\n[{index}/6] {stage}: завершено за {record['elapsed_seconds'] / 60:.1f} хв", flush=True)
    print(f"\nУсі 6 етапів завершено. Результати: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
