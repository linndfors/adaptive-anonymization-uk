"""Check Git-visible source files for generated data, binaries and credentials."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIRECTORIES = {"artifacts", "results", "notebooks", "models", "outputs", "reports",
                       ".local-notes", ".uv-cache", ".venv", "__pycache__"}
DATA_FILES = {"data/README.md"}
SOURCE_SUFFIXES = {".py", ".md", ".json", ".toml", ".lock", ".css", ".txt", ".yml", ".yaml", ".in"}
SPECIAL_FILES = {".gitignore", ".gitattributes", ".python-version", "LICENSE", "NOTICE"}
SECRET_PATTERNS = [re.compile(pattern) for pattern in (
    r"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{24,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"AKIA[0-9A-Z]{16}",
)]


def inspect(path, root):
    relative = path.relative_to(root)
    name = relative.as_posix()
    errors = []
    if path.is_symlink():
        return [f"Symlink in source submission: {name}"]
    if PRIVATE_DIRECTORIES.intersection(relative.parts) or any(p.startswith('.venv') for p in relative.parts):
        errors.append(f"Generated/private directory: {name}")
    if path.name.startswith('.env') or path.name in {"secrets.toml", ".DS_Store"}:
        errors.append(f"Local configuration or credential file: {name}")
    if relative.parts[0] == 'data' and name not in DATA_FILES:
        errors.append(f"Dataset or generated data: {name}")
    if path.suffix not in SOURCE_SUFFIXES and path.name not in SPECIAL_FILES:
        errors.append(f"Unexpected non-source file: {name}")
    if path.stat().st_size > 2_000_000:
        errors.append(f"File exceeds the 2 MB source limit: {name}")
        return errors
    try:
        content = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        errors.append(f"Binary content: {name}")
    else:
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"Credential pattern: {name}")
    return errors


def candidate_files(root):
    result = subprocess.run(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        check=True, capture_output=True,
    )
    paths = {root / name.decode('utf-8') for name in result.stdout.split(b'\0') if name}
    return sorted(path for path in paths if path.is_file() or path.is_symlink())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = candidate_files(root)
    errors = [error for path in paths for error in inspect(path, root)]
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'source_files_checked': len(paths), 'bytes': sum(p.stat().st_size for p in paths),
                      'status': 'passed'}))


if __name__ == '__main__':
    main()
