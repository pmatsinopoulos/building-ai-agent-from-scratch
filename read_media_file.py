import base64
import os
from pathlib import Path
from typing import Any, cast

import pymupdf as fitz
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from tools import tool

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]
AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".webm"]
PDF_EXTENSIONS = [".pdf"]


@tool
def read_media_file(file_path: str, query: str) -> str:
    """Analyze an image, audio, or PDF file using LLM."""

    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _analyze_image(file_path, query)
    elif ext in AUDIO_EXTENSIONS:
        return _analyze_audio(file_path, query)
    elif ext in PDF_EXTENSIONS:
        return _analyze_pdf(file_path, query)
    else:
        return f"Unsupported media format: {ext}"


def _analyze_image(file_path: str, query: str) -> str:
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip(".")
    media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data: {media_type};base64,{image_data}"},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content or ""


def _analyze_audio(file_path: str, query: str) -> str:
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip(".")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": ext},
                    },
                ],
            }
        ],
    )
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=messages,
    )

    return response.choices[0].message.content or ""


def _analyze_pdf(file_path: str, query: str) -> str:
    doc = fitz.open(file_path)  # type: ignore[no-untyped-call]

    # Extract text for context
    text_content = ""
    for page in doc:  # type: ignore[attr-defined]
        text_content += cast(str, page.get_text())  # pyright: ignore[reportUnknownMemberType]

    # convert pages to images
    images: list[str] = []
    for page in doc[:5]:  # first 5 pages
        pix = page.get_pixmap(  # pyright: ignore[reportUnknownMemberType]
            matrix=fitz.Matrix(2, 2),  # type: ignore[no-untyped-call]
        )
        img_bytes = cast(bytes, pix.tobytes("png"))  # type: ignore[no-untyped-call]
        images.append(base64.b64encode(img_bytes).decode("utf-8"))

    # Build content with text and images
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"{query}\n\nExtracted text:\n{text_content[:3000]}"},
    ]

    for img_b64 in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = cast(
        list[ChatCompletionMessageParam],
        [{"role": "user", "content": content}],
    )
    response = client.chat.completions.create(model="gpt-4o", messages=messages)

    return response.choices[0].message.content or ""
