import type { RecommendationResponse } from "../types";

export async function fetchRecommendations(
  userId: string,
  topK: number = 10
): Promise<RecommendationResponse> {
  const apiUrl = import.meta.env.VITE_API_URL || "";
  const url = `${apiUrl}/api/recommendations/${userId}?top_k=${topK}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return await response.json();
}
