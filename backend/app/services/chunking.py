
import re


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 100
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    # Normalize line endings and split into paragraphs
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current_paragraphs = []
    current_length = 0

    for paragraph in paragraphs:

        # Handle paragraphs larger than the target chunk size
        if len(paragraph) > chunk_size:

            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_length = 0

            start = 0

            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())
                start += chunk_size - overlap

            continue

        added_length = len(paragraph) + (
            2 if current_paragraphs else 0
        )

        if current_length + added_length <= chunk_size:

            current_paragraphs.append(paragraph)
            current_length += added_length

        else:
            # Save the current chunk
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))

            # Carry forward complete paragraphs for context
            overlap_paragraphs = []
            overlap_length = 0

            for previous in reversed(current_paragraphs):
                if overlap_length + len(previous) > overlap:
                    break

                overlap_paragraphs.insert(0, previous)
                overlap_length += len(previous)

            current_paragraphs = overlap_paragraphs + [paragraph]
            current_length = sum(
                len(p) for p in current_paragraphs
            ) + 2 * (len(current_paragraphs) - 1)

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks
