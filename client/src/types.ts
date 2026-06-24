export interface Recommendation {
  item_id: string;
  score: number;
  rank: number;
}

export interface RecommendationResponse {
  user_id: string;
  recommendations: Recommendation[];
}

export interface HealthResponse {
  status: string;
}

export interface PipelineJob {
  job_id: string;
  step: string;
  status: "pending" | "running" | "completed" | "failed";
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface PipelineTriggerResponse {
  job_id: string;
  message: string;
}
