
import base64
import secrets
import uuid
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from app.config import settings
from app.agent import graph


# --------------------------------------------------
# PATHS AND APPLICATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Medical Agent Educational API",
    version="1.0.0",
    description=(
        "A medical education assistant using Gemini, "
        "Pinecone, and optional Tavily web search."
    ),
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.allowed_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Conversation-ID",
    ],
)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static",
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )

    history: list[dict[str, str]] = Field(
        default_factory=list,
        max_length=10,
    )

    image_data: str | None = None


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def get_gemini_client():
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is missing from your .env file.",
        )

    return genai.Client(
        api_key=settings.gemini_api_key,
    )


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def auth(authorization: str | None):
    if not settings.api_token:
        return

    expected = f"Bearer {settings.api_token}"

    if not secrets.compare_digest(
        authorization or "",
        expected,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API token.",
        )


# --------------------------------------------------
# IMAGE ANALYSIS USING GEMINI
# --------------------------------------------------

def inspect_image(data: str | None) -> str:
    if not data:
        return ""

    try:
        header, payload = data.split(",", 1)

        mime = header.split(";")[0].split(":")[1]

        raw = base64.b64decode(
            payload,
            validate=True,
        )

        if mime not in (
            "image/jpeg",
            "image/png",
            "image/webp",
        ):
            raise ValueError(
                "Only JPEG, PNG, and WebP images are supported."
            )

        if len(raw) > 5_000_000:
            raise ValueError(
                "Image size must not exceed 5 MB."
            )

    except (ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=settings.gemini_vision_model,
            contents=[
                (
                    "Describe only objectively visible, "
                    "medically relevant features in this image. "
                    "Do not provide a definitive diagnosis, "
                    "prescribe treatment, or claim that the "
                    "image proves or excludes a disease. "
                    "Mention uncertainty when appropriate."
                ),
                types.Part.from_bytes(
                    data=raw,
                    mime_type=mime,
                ),
            ],
        )

        return (
            response.text
            or "The image could not be described reliably."
        )

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Gemini image analysis error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini image analysis failed. "
                "Check your API key, model name, and API quota."
            ),
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = ROOT / "templates" / "index.html"

    if not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend file templates/index.html was not found.",
        )

    return html_path.read_text(
        encoding="utf-8",
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(
            settings.gemini_api_key
        ),
        "pinecone_configured": bool(
            settings.pinecone_api_key
        ),
        "tavily_configured": bool(
            settings.tavily_api_key
        ),
    }


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat")
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(
        default=None,
    ),
    conversation_id: str | None = Header(
        default=None,
        alias="X-Conversation-ID",
    ),
):
    auth(authorization)

    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=503,
            detail="Configure GEMINI_API_KEY in your .env file.",
        )

    if not settings.pinecone_api_key:
        raise HTTPException(
            status_code=503,
            detail="Configure PINECONE_API_KEY in your .env file.",
        )

    # Keep only valid conversation messages.
    history = [
        item
        for item in payload.history
        if item.get("role") in (
            "user",
            "assistant",
        )
        and isinstance(
            item.get("content"),
            str,
        )
    ][-10:]

    # Reuse the conversation ID to preserve checkpoint memory.
    thread_id = conversation_id or str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        # Analyze an optional image using Gemini.
        image_description = inspect_image(
            payload.image_data,
        )

        # Execute the LangGraph agent.
        state = graph.invoke(
            {
                "question": payload.message,
                "history": history,
                "image_description": image_description,
                "kb": [],
                "web": [],
                "answer": "",
                "route": "",
            },
            config=config,
        )

        # Prepare source information.
        sources = []

        for item in state.get("kb", []):
            sources.append(
                {
                    "type": "pdf",
                    "page": item.get("page"),
                    "score": item.get("score"),
                    "source": item.get("source"),
                }
            )

        for item in state.get("web", []):
            sources.append(
                {
                    "type": "web",
                    "url": item.get("url"),
                    "title": item.get("title"),
                }
            )

        # Report which integrations were used.
        tools_used = [
            "gemini",
            "pinecone",
        ]

        if (
            state.get("route") == "web"
            and settings.tavily_api_key
        ):
            tools_used.append("tavily")

        if image_description:
            tools_used.append("gemini-vision")

        return {
            "answer": state.get("answer", ""),
            "sources": sources,
            "tools": tools_used,
            "conversation_id": thread_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Chat error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "AI service failed. Check the VS Code terminal "
                "for details. Verify Gemini credentials, model "
                "availability, Pinecone configuration, and connectivity."
            ),
        )


# --------------------------------------------------
# AUDIO TRANSCRIPTION USING GEMINI
# --------------------------------------------------

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(
        default=None,
    ),
):
    auth(authorization)

    # Read at most 10 MB plus one byte to detect oversized files.
    data = await file.read(10_000_001)

    if len(data) > 10_000_000:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds 10 MB.",
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The audio file is empty.",
        )

    mime_type = file.content_type or "audio/webm"

    if not mime_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported audio file.",
        )

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=[
                (
                    "Transcribe the spoken words in this audio "
                    "as accurately as possible. Return only "
                    "the transcription. Do not answer questions "
                    "contained in the audio."
                ),
                types.Part.from_bytes(
                    data=data,
                    mime_type=mime_type,
                ),
            ],
        )

        return {
            "text": response.text or "",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            "Gemini transcription error:",
            repr(exc),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Audio transcription failed. Check the audio "
                "format, Gemini model support, and API quota."
            ),
        )
