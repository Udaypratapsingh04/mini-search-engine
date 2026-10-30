"""FastAPI backend.

Run locally:   python -m uvicorn app.main:app --reload
Swagger UI:    http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.service import SearchService

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_QUERY_LEN = 200
MAX_K = 50


class Hit(BaseModel):
    doc_id: int
    score: float
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    k: int
    count: int
    took_ms: float
    did_you_mean: Optional[str] = None
    results: List[Hit]


class SuggestResponse(BaseModel):
    prefix: str
    suggestions: List[str]
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    docs: int
    vocab_words: int


def create_app(data_dir=None, preprocess_fn=None, service=None) -> FastAPI:
    """App factory. `service` lets tests inject a ready-made SearchService."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the index ONCE at startup, never per request.
        app.state.service = service or SearchService.from_data_dir(data_dir, preprocess_fn)
        yield

    app = FastAPI(title="Mini Search Engine", version="1.0.0", lifespan=lifespan)

    def get_service(request: Request) -> SearchService:
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Search index is not loaded yet")
        return svc

    @app.get("/search", response_model=SearchResponse)
    def search(
        request: Request,
        q: str = Query(..., min_length=1, max_length=MAX_QUERY_LEN,
                       description="Ranked, boolean (AND/OR/NOT) or \"phrase\" query"),
        k: int = Query(10, ge=1, le=MAX_K, description="Number of results"),
    ):
        if not q.strip():
            raise HTTPException(status_code=422, detail="q must not be blank")
        return get_service(request).search(q, k)

    @app.get("/suggest", response_model=SuggestResponse)
    def suggest(
        request: Request,
        prefix: str = Query(..., min_length=1, max_length=50),
        limit: int = Query(5, ge=1, le=20),
    ):
        return get_service(request).suggest(prefix, limit)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        return get_service(request).health()

    @app.get("/", include_in_schema=False)
    def home():
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return {"message": "Mini Search Engine API. See /docs"}

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
