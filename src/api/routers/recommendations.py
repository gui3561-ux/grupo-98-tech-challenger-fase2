from fastapi import APIRouter, HTTPException, Request

from api.schemas import Recommendation, RecommendationResponse

router = APIRouter(prefix="/api")


@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request, user_id: str, top_k: int = 10
) -> RecommendationResponse:
    inference = request.app.state.inference

    if not inference.is_ready:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Run training first."
        )

    results = inference.recommend(user_id, top_k=top_k)

    if results is None:
        raise HTTPException(status_code=404, detail="User not found")

    return RecommendationResponse(
        user_id=user_id,
        recommendations=[Recommendation(**r) for r in results],
    )
