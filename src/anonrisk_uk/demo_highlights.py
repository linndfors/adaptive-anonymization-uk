"""Render exact applied spans; never infer text edits from character similarity."""

import html


def _escape(value):
    return html.escape(value, quote=True)


def _mark(value, label, color):
    return (f'<mark title="{_escape(label)}" style="background:{color};color:#172033;'
            'border-radius:3px;box-decoration-break:clone;-webkit-box-decoration-break:clone">'
            f'{_escape(value)}</mark>')


def highlighted_pair(original, output, edits):
    """Use the executor's offsets for both panes and verify the reconstruction.

    Only actual replacements are highlighted. Untouched skills stay uncoloured.
    A mismatch fails explicitly instead of displaying a plausible but false diff.
    """
    left, right, rebuilt, cursor = [], [], [], 0
    for edit in sorted((e for e in edits if e["applied"]), key=lambda e: e["start"]):
        start, end = edit.get("render_start", edit["start"]), edit.get("render_end", edit["end"])
        if not 0 <= cursor <= start < end <= len(original):
            raise ValueError("Invalid or overlapping edit offsets")
        if original[start:end] != edit.get("render_text", edit["text"]):
            raise ValueError("Edit does not match original text")
        unchanged = original[cursor:start]
        replacement = edit["replacement"]
        left.extend((_escape(unchanged), _mark(original[start:end], edit["label"], "#fecdd3")))
        right.extend((_escape(unchanged), _mark(replacement, edit["label"], "#bbf7d0") if replacement else ""))
        rebuilt.extend((unchanged, replacement))
        cursor = end
    tail = original[cursor:]
    rebuilt.append(tail)
    if "".join(rebuilt) != output:
        raise ValueError("Edits do not reconstruct the released text")
    left.append(_escape(tail))
    right.append(_escape(tail))
    return "".join(left), "".join(right)
