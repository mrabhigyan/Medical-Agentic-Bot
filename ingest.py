
import argparse
import hashlib
from pathlib import Path

import fitz

from app.knowledge import ensure_index, embed


def chunks(
    text: str,
    size: int = 1100,
    overlap: int = 180,
) -> list[str]:
    """Split text into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    step = size - overlap

    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if len(text[i:i + size]) > 80
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a medical PDF using Gemini and Pinecone."
    )

    parser.add_argument(
        "--pdf",
        default="data/medical_book.pdf",
        help="Path to your medical PDF.",
    )

    args = parser.parse_args()
    pdf = Path(args.pdf)

    if not pdf.exists():
        raise SystemExit(
            f"PDF not found: {pdf}\n"
            "Check that the file exists at the specified path."
        )

    index = ensure_index()

    with fitz.open(pdf) as doc:
        batch = []
        total_pages = len(doc)

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()

            for part, content in enumerate(
                chunks(page_text),
                start=1,
            ):
                key = hashlib.sha256(
                    (
                        f"{pdf.name}:"
                        f"{page_number}:"
                        f"{part}:"
                        f"{content}"
                    ).encode("utf-8")
                ).hexdigest()

                metadata = {
                    "page": page_number,
                    "text": content,
                    "file": pdf.name,
                }

                batch.append((key, content, metadata))

                if len(batch) >= 50:
                    upload_batch(index, batch)
                    batch = []

            if page_number % 10 == 0 or page_number == total_pages:
                print(
                    f"Processed pages: {page_number}/{total_pages}",
                    flush=True,
                )

        if batch:
            upload_batch(index, batch)

    print("Medical PDF ingestion complete!")


def upload_batch(index, batch):
    """Embed and upload one batch of text chunks."""

    print(f"Embedding and uploading {len(batch)} chunks...")

    vectors = embed([item[1] for item in batch])

    records = [
        {
            "id": item[0],
            "values": vector,
            "metadata": item[2],
        }
        for item, vector in zip(batch, vectors)
    ]

    index.upsert(vectors=records)

    print(f"Uploaded {len(records)} chunks.", flush=True)


if __name__ == "__main__":
    main()
