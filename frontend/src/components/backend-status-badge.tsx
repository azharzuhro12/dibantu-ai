"use client";

/** Backend connection badge in the top bar, backed by GET /health. */

import { useBackendHealth } from "@/hooks/use-backend-health";
import { API_BASE_URL } from "@/lib/api";

export function BackendStatusBadge() {
  const health = useBackendHealth();

  return (
    <div
      aria-live="polite"
      className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium shadow-sm"
      title={
        health.state === "offline" && health.error
          ? `${health.error} (API: ${API_BASE_URL})`
          : `API: ${API_BASE_URL}`
      }
    >
      {health.state === "online" ? (
        <span className="relative flex h-2 w-2" aria-hidden="true">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
        </span>
      ) : health.state === "offline" ? (
        <span className="h-2 w-2 rounded-full bg-rose-500" aria-hidden="true" />
      ) : (
        <span className="h-2 w-2 animate-pulse rounded-full bg-slate-400" aria-hidden="true" />
      )}

      <span className="text-slate-700">
        {health.state === "online" && health.version
          ? `Connected · v${health.version}`
          : health.state === "offline"
            ? "Backend offline"
            : "Connecting…"}
      </span>
    </div>
  );
}
