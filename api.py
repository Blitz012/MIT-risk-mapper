"""Citable REST API for the MIT AI Risk Mapper.

This module exposes the vector mapping layer over HTTP so that the tool can be
cited and reused programmatically, not only through the Streamlit interface. A
researcher or another service can POST an AI project description and receive the
top matching MIT risk domains together with their cosine similarity scores.

The design mirrors the rest of the codebase:

    The heavy embedding model is loaded exactly once, at application startup, and
    reused across requests. This keeps each request fast because the model is the
    expensive part, not the inference.

    Request and response shapes are declared with Pydantic models, which gives us
    automatic validation, clear error messages, and a self documenting OpenAPI
    schema at /docs without any extra work.

    The mapper itself is untouched. This API is a thin, well typed transport layer
    on top of the existing RiskMapper, so the science stays in one place and stays
    citable.

Run locally with:

    uvicorn api:app --reload

Then open http://localhost:8000/docs for interactive documentation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nlp_mapper import RiskMapper

# The default number of matches to return when the caller does not specify one.
DEFAULT_TOP_N = 3
# Guard rails on how many matches a single request may ask for, so a caller
# cannot request an unbounded or nonsensical number of results.
MIN_TOP_N = 1
MAX_TOP_N = 10


class AnalyzeRequest(BaseModel):
    """The JSON payload a client sends to request a risk analysis.

    Attributes:
        description: The free text description of the AI project to analyse.
        top_n: How many of the closest risk subdomains to return, ranked best
            match first. Bounded to a sensible range.
    """

    description: str = Field(
        ...,
        min_length=1,
        description="Free text description of the AI project to analyse.",
        examples=[
            "A health app that uses AI to predict patient outcomes from records."
        ],
    )
    top_n: int = Field(
        DEFAULT_TOP_N,
        ge=MIN_TOP_N,
        le=MAX_TOP_N,
        description="Number of top risk matches to return, ranked best first.",
    )


class RiskMatch(BaseModel):
    """A single ranked risk match returned to the client.

    Attributes:
        rank: The 1 based position of this match, where 1 is the closest.
        domain: The MIT risk domain, for example 'Privacy & Security'.
        subdomain: The specific subdomain within that domain.
        definition: The taxonomy definition of the subdomain, for context.
        score: The cosine similarity between the description and this subdomain,
            on a scale where higher means more similar.
    """

    rank: int
    domain: str
    subdomain: str
    definition: str
    score: float


class AnalyzeResponse(BaseModel):
    """The JSON response returned for a successful analysis.

    Attributes:
        description: Echo of the description that was analysed, so the result is
            self contained and citable on its own.
        count: The number of matches returned.
        matches: The ranked list of risk matches.
    """

    description: str
    count: int
    matches: List[RiskMatch]


# A module level holder for the single shared mapper instance. It is populated
# during the startup lifespan event and read by the request handlers.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model once at startup and share it across requests.

    FastAPI calls this context manager when the application starts and again when
    it shuts down. Loading the RiskMapper here, rather than per request, means the
    expensive model load happens a single time for the life of the process.
    """
    _state["mapper"] = RiskMapper()
    yield
    # Nothing to tear down: the mapper holds only in memory state that the
    # interpreter reclaims on exit.
    _state.clear()


app = FastAPI(
    title="MIT AI Risk Mapper API",
    description=(
        "Maps an AI project description to the closest MIT AI Risk taxonomy "
        "domains and subdomains using sentence embedding cosine similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_mapper() -> RiskMapper:
    """Return the shared mapper, or signal that the service is not ready.

    Raises:
        HTTPException: With status 503 if the model has not finished loading,
            which should only happen if a request arrives before startup.
    """
    mapper = _state.get("mapper")
    if mapper is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded yet, please retry shortly."
        )
    return mapper


@app.get("/health")
def health() -> dict:
    """Lightweight liveness and readiness check.

    Returns a simple status object reporting whether the model is loaded. This is
    useful for uptime monitors and for confirming the service is ready to serve.
    """
    return {"status": "ok", "model_loaded": _state.get("mapper") is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyse one AI project description and return its top risk matches.

    Args:
        request: The validated request payload holding the description and the
            desired number of matches.

    Returns:
        An AnalyzeResponse with the ranked matches and their similarity scores.

    Raises:
        HTTPException: With status 400 if the description is blank once trimmed.
    """
    description = request.description.strip()
    if not description:
        raise HTTPException(
            status_code=400, detail="Description must not be empty or whitespace."
        )

    mapper = get_mapper()
    ranked = mapper.find_top_risks(description, n=request.top_n)

    matches = [
        RiskMatch(
            rank=rank,
            domain=entry.domain,
            subdomain=entry.subdomain,
            definition=entry.definition,
            score=float(score),
        )
        for rank, (entry, score) in enumerate(ranked, start=1)
    ]

    return AnalyzeResponse(
        description=description,
        count=len(matches),
        matches=matches,
    )
