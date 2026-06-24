import { useState } from "react";
import { fetchRecommendations } from "./api/recommendations";
import { HealthStatus } from "./components/HealthStatus";
import { PipelinePanel } from "./components/PipelinePanel";
import { RecommendationCard } from "./components/RecommendationCard";
import { UserSearch } from "./components/UserSearch";
import type { RecommendationResponse } from "./types";

export default function App() {
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSearch(userId: string) {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchRecommendations(userId);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-10">
      <div className="mx-auto max-w-3xl">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900">
            Recommender Test Client
          </h1>
          <p className="mt-2 text-gray-600">
            Enter a user ID to fetch product recommendations
          </p>
          <div className="mt-3 flex justify-center">
            <HealthStatus />
          </div>
        </header>

        <UserSearch onSearch={handleSearch} isLoading={isLoading} />

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-8">
            <h2 className="mb-4 text-lg font-semibold text-gray-800">
              Recommendations for user:{" "}
              <span className="text-blue-600">{result.user_id}</span>
            </h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {result.recommendations.map((rec) => (
                <RecommendationCard key={rec.item_id} recommendation={rec} />
              ))}
            </div>
          </div>
        )}

        <div className="mt-10">
          <PipelinePanel />
        </div>
      </div>
    </div>
  );
}
