import math

def chunk_text(text: str, max_chars: int = 800):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to split at sentence boundary
        seg = text[start:end]
        # extend to next punctuation if available
        if end < len(text):
            # try to find last sentence end
            last_dot = seg.rfind('.')
            last_nl = seg.rfind('\n')
            cut = max(last_dot, last_nl)
            if cut > int(max_chars*0.5):
                end = start + cut + 1
        parts.append(text[start:end].strip())
        start = end
    return parts
