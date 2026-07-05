import { useState } from "react";
import {
  triggerPipeline,
  listJobs,
  getJob,
  reloadModel,
} from "../api/pipeline";
import type { PipelineJob } from "../types";

const STAGES = [
  { key: "preprocessing", label: "Preprocessing" },
  { key: "feature-engineering", label: "Feature Engineering" },
  { key: "training", label: "Training" },
  { key: "evaluation", label: "Evaluation" },
] as const;

export function PipelinePanel() {
  const [jobs, setJobs] = useState<PipelineJob[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleTrigger(
    stage: (typeof STAGES)[number]["key"]
  ) {
    console.log("stage", stage);
    setError(null);
    setMessage(null);
    setLoading(true);
    try {
      const res = await triggerPipeline(stage);
      setMessage(`Triggered ${stage} — Job ID: ${res.job_id}`);
      await refreshJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger stage");
    } finally {
      setLoading(false);
    }
  }

  async function refreshJobs() {
    try {
      const res = await listJobs();
      setJobs(res);
    } catch {
      setJobs([]);
    }
  }

  async function handleRefreshJob(jobId: string) {
    try {
      const updated = await getJob(jobId);
      setJobs((prev) =>
        prev.map((j) => (j.job_id === jobId ? updated : j))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to get job");
    }
  }

  async function handleReloadModel() {
    setError(null);
    setMessage(null);
    setLoading(true);
    try {
      const res = await reloadModel();
      setMessage(res.message || "Model reloaded successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reload model");
    } finally {
      setLoading(false);
    }
  }

  const statusColor = (status: PipelineJob["status"]) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800";
      case "running":
        return "bg-yellow-100 text-yellow-800";
      case "failed":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-lg font-semibold text-gray-800">
        Pipeline Controls
      </h2>

      <div className="mb-4 flex flex-wrap gap-2">
        {STAGES.map((stage) => (
          <button
            key={stage.key}
            onClick={() => handleTrigger(stage.key)}
            disabled={loading}
            className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:opacity-50"
          >
            {stage.label}
          </button>
        ))}
        <button
          onClick={handleReloadModel}
          disabled={loading}
          className="rounded-md bg-amber-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-amber-700 disabled:opacity-50"
        >
          Reload Model
        </button>
      </div>

      <button
        onClick={refreshJobs}
        className="mb-4 text-sm text-blue-600 underline hover:text-blue-800"
      >
        Refresh Jobs
      </button>

      {message && (
        <div className="mb-3 rounded border border-green-200 bg-green-50 p-3 text-sm text-green-700">
          {message}
        </div>
      )}
      {error && (
        <div className="mb-3 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {jobs?.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b text-gray-600">
              <tr>
                <th className="pb-2 pr-4">Job ID</th>
                <th className="pb-2 pr-4">Stage</th>
                <th className="pb-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id} className="border-b last:border-0">
                  <td
                    className="py-2 pr-4 font-mono text-xs"
                    style={{ overflow: "hidden" }}
                  >
                    {job.job_id.slice(0, 8)}
                  </td>
                  <td className="py-2 pr-4">{job.step}</td>
                  <td className="py-2 pr-4">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-semibold ${statusColor(job.status)}`}
                    >
                      {job.status}
                    </span>
                    <button
                      onClick={() => handleRefreshJob(job.job_id)}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      refresh
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
