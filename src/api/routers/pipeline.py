from fastapi import APIRouter, HTTPException, Request

from api.schemas import JobCreated, JobStatusResponse

router = APIRouter(prefix="/api/pipeline")


def _trigger_step(request: Request, step: str, target: callable) -> JobCreated:
    job_manager = request.app.state.job_manager

    if job_manager.has_running_job(step):
        raise HTTPException(status_code=409, detail=f"{step} is already running")

    job_id = job_manager.create_job(step)
    job_manager.run_in_background(job_id, target)
    return JobCreated(job_id=job_id, message=f"{step} job started")


@router.post("/preprocessing", status_code=202, response_model=JobCreated)
async def trigger_preprocessing(request: Request) -> JobCreated:
    from preprocessing.cli import main

    return _trigger_step(request, "preprocessing", main)


@router.post("/feature-engineering", status_code=202, response_model=JobCreated)
async def trigger_feature_engineering(request: Request) -> JobCreated:
    from preprocessing.feature_cli import main

    return _trigger_step(request, "feature_engineering", main)


@router.post("/training", status_code=202, response_model=JobCreated)
async def trigger_training(request: Request) -> JobCreated:
    from training.cli import main

    return _trigger_step(request, "training", main)


@router.post("/evaluation", status_code=202, response_model=JobCreated)
async def trigger_evaluation(request: Request) -> JobCreated:
    from evaluation.cli import main

    return _trigger_step(request, "evaluation", main)


@router.post("/reload-model")
async def reload_model(request: Request) -> dict:
    inference = request.app.state.inference
    try:
        inference.load_model()
        return {"message": "Model reloaded successfully"}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Model file not found. Run training first."
        )


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(request: Request) -> list[JobStatusResponse]:
    job_manager = request.app.state.job_manager
    jobs = job_manager.list_jobs()
    return [
        JobStatusResponse(
            job_id=j.job_id,
            step=j.step,
            status=j.status,
            started_at=j.started_at,
            completed_at=j.completed_at,
            error=j.error,
        )
        for j in jobs
    ]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(request: Request, job_id: str) -> JobStatusResponse:
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        step=job.step,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )
