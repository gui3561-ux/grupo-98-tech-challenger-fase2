from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.pipeline import router as pipeline_router
from api.routers.recommendations import router as recommendations_router
from api.services.inference import InferenceService
from api.services.job_manager import JobManager
from utils import Settings, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    inference = InferenceService(settings)

    try:
        inference.load_model()
    except FileNotFoundError:
        logger.warning("Model not found — inference disabled until training completes")

    app.state.inference = inference
    app.state.job_manager = JobManager()
    app.state.settings = settings
    yield


app = FastAPI(title="E-commerce Recommender API", lifespan=lifespan)

settings = Settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router)
app.include_router(pipeline_router)


@app.get("/api/health")
async def health():
    return {"status": "healthy", "model_loaded": app.state.inference.is_ready}


def serve() -> None:
    import uvicorn

    s = Settings()
    uvicorn.run(
        "api.app:app",
        host=s.api_host,
        port=s.api_port,
        reload=s.app_env == "development",
    )


if __name__ == "__main__":
    serve()
