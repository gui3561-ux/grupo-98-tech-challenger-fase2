import type { Recommendation } from "../types";

interface Props {
  recommendation: Recommendation;
}

export function RecommendationCard({ recommendation }: Props) {
  const { item_id, score, rank } = recommendation;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm transition-shadow hover:shadow-md">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-gray-500">#{rank}</span>
        <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-semibold text-blue-800">
          {(score * 100).toFixed(1)}%
        </span>
      </div>
      <p className="text-lg font-semibold text-gray-900">{item_id}</p>
      <div className="mt-2 h-1.5 w-full rounded-full bg-gray-200">
        <div
          className="h-1.5 rounded-full bg-blue-600"
          style={{ width: `${score * 100}%` }}
        />
      </div>
    </div>
  );
}
