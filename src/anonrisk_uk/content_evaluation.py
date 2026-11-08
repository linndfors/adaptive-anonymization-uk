"""Post-hoc semantic diagnostics, independent of the anonymization objective.

Cosines measure embedding similarity, NOT entailment or factual correctness.
All token windows are encoded; long resumes are never silently truncated.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .workua import GROUP_LABELS

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PROFESSIONAL_GROUPS = {"title", "exp_position", "skills", "exp_resp", "edu_faculty"}
HEADINGS = re.compile(r"(?i)(?<!\w)(" + "|".join(re.escape(v) for v in
                      sorted(GROUP_LABELS.values(), key=len, reverse=True)) + r")\s*:\s*")


def professional_view(text):
    """Extract public professional fields, excluding identity/demographic headers.

    A format-specific diagnostic for WorkUA serialization, not a general parser.
    Typed masks carry no professional meaning and are excluded from the view.
    """
    groups = {value.casefold(): group for group, value in GROUP_LABELS.items()}
    matches = list(HEADINGS.finditer(text))
    values = []
    for i, match in enumerate(matches):
        if groups[match[1].casefold()] not in PROFESSIONAL_GROUPS:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = re.sub(r"\[[^\]]*\]", " ", text[match.end():end])
        value = re.sub(r"\s+", " ", value).strip(" .;,\n")
        if re.search(r"\w", value):
            values.append(value)
    return ". ".join(values)


def token_windows(ids, capacity):
    if capacity <= 0:
        raise ValueError("Window capacity must be positive")
    return [ids[start:start + capacity] for start in range(0, len(ids), capacity)]


class SemanticEncoder:
    def __init__(self, model_dir, batch_size=24):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(Path(model_dir), local_files_only=True).eval().to("cpu")
        self.capacity = 128 - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.dim = self.model.config.hidden_size
        self.cache = {}

    def encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        windows, owners, weights = [], [], []
        for text in missing:
            ids = self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            for window in token_windows(ids, self.capacity):
                windows.append(self.tokenizer.prepare_for_model(window, return_attention_mask=True,
                                                                return_token_type_ids=False))
                owners.append(text)
                weights.append(len(window))
            self.cache[text] = {"vector": np.zeros(self.dim, dtype=np.float32), "chunks": []}
        for start in range(0, len(windows), self.batch_size):
            batch = self.tokenizer.pad(windows[start:start + self.batch_size], padding=True, return_tensors="pt")
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = self.torch.nn.functional.normalize(vectors, dim=1).numpy()
            for j, vector in enumerate(vectors, start):
                item = self.cache[owners[j]]
                item["vector"] += vector * weights[j]
                item["chunks"].append(vector)
        for text in missing:
            item = self.cache[text]
            vector = item["vector"]
            item["vector"] = vector / max(float(np.linalg.norm(vector)), 1e-12)
            item["chunks"] = np.asarray(item["chunks"], dtype=np.float32).reshape(-1, self.dim)
        return [self.cache[text] for text in texts]


def semantic_pair(original, output):
    """An empty source is unscorable; an emptied output loses all source content."""
    a, b = original["chunks"], output["chunks"]
    if not len(a):
        return {"cosine": None, "chunk_recall": None, "chunk_precision": None}
    if not len(b):
        return {"cosine": 0.0, "chunk_recall": 0.0, "chunk_precision": 0.0}
    similarity = a @ b.T
    return {"cosine": float(original["vector"] @ output["vector"]),
            "chunk_recall": float(similarity.max(axis=1).mean()),
            "chunk_precision": float(similarity.max(axis=0).mean())}
