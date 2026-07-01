import type {
  HealthResponse,
  PipelineJob,
  PipelineTriggerResponse,
} from "../types";

const getApiUrl = () => import.meta.env.VITE_API_URL || "";

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${getApiUrl()}/api/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }
  return response.json();
}

type PipelineStage =
  | "preprocessing"
  | "feature_engineering"
  | "training"
  | "evaluation";

export async function triggerPipeline(
  stage: PipelineStage
): Promise<PipelineTriggerResponse> {
  const response = await fetch(`${getApiUrl()}/api/pipeline/${stage}`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to trigger ${stage}: ${response.status}`);
  }
  return response.json();
}

export async function listJobs(): Promise<PipelineJob[]> {
  const response = await fetch(`${getApiUrl()}/api/pipeline/jobs`);
  if (!response.ok) {
    throw new Error(`Failed to list jobs: ${response.status}`);
  }
  return response.json();
}

export async function getJob(jobId: string): Promise<PipelineJob> {
  const response = await fetch(`${getApiUrl()}/api/pipeline/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? `Job ${jobId} not found`
        : `Failed to get job: ${response.status}`
    );
  }
  return response.json();
}

export async function reloadModel(): Promise<{ message: string }> {
  const response = await fetch(`${getApiUrl()}/api/pipeline/reload-model`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to reload model: ${response.status}`);
  }
  return response.json();
}
