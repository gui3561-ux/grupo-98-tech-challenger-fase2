from datetime import datetime

from pydantic import BaseModel


class Recommendation(BaseModel):
    item_id: str
    score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[Recommendation]


class JobCreated(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    step: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
