import { useState, useEffect } from "react";
import { checkHealth } from "../api/pipeline";

export function HealthStatus() {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${
          healthy === null
            ? "bg-gray-400"
            : healthy
              ? "bg-green-500"
              : "bg-red-500"
        }`}
      />
      <span className="text-gray-600">
        API: {healthy === null ? "checking..." : healthy ? "online" : "offline"}
      </span>
    </div>
  );
}
