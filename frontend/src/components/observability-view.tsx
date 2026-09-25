"use client";

/**
 * Agent run tracer (Step 15), wired to the real backend:
 *   GET /api/observability/runs
 *   GET /api/observability/runs/{run_id}/events
 *
 * Master-detail: filter recent runs (status, source, owner), then open
 * one run to inspect its event timeline — LLM calls, tool calls, memory,
 * RAG, approvals — with durations and safe metadata only. Records store
 * no prompts, no responses, and no secrets by construction.
 */

import { useCallback, useEffect, useState } from "react";

import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { IconAlert, IconPulse, IconRefresh } from "@/components/icons";
import {
  listAgentRunEvents,
  listAgentRuns,
  type AgentEvent,
  type AgentRun,
  type AgentRunStatus,
} from "@/lib/api";
import { formatBackendTimestamp } from "@/lib/format";

const STATUS_STYLES: Record<string, string> = {
  running: "border-amber-200 bg-amber-50 text-amber-700",
  completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
};

const EVENT_TYPE_STYLES: Record<string, string> = {
  RUN: "border-slate-300 bg-slate-100 text-slate-700",
  LLM: "border-indigo-200 bg-indigo-50 text-indigo-700",
  TOOL: "border-sky-200 bg-sky-50 text-sky-700",
  MEMORY: "border-emerald-200 bg-emerald-50 text-emerald-700",
  RAG: "border-amber-200 bg-amber-50 text-amber-700",
  APPROVAL: "border-violet-200 bg-violet-50 text-violet-700",
};

const STATUS_FILTERS: { value: "" | AgentRunStatus; label: string }[] = [
  { value: "", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
];

function formatDuration(ms: number | null): string {
  if (ms === null) {
    return "—";
  }
  if (ms < 1000) {
    return `${ms} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

function eventSummary(event: AgentEvent): string {
  const meta = event.metadata ?? {};
  const tool = typeof meta.tool_name === "string" ? meta.tool_name : null;
  const model = typeof meta.model === "string" ? meta.model : null;
  const parts = [event.event_name];
  if (tool) {
    parts.push(tool);
  } else if (model) {
    parts.push(model);
  }
  return parts.join(" · ");
}

function eventMetaLine(event: AgentEvent): string | null {
  const meta = event.metadata ?? {};
  const entries = Object.entries(meta)
    .filter(([, value]) => value !== null && typeof value !== "object")
    .map(([key, value]) => `${key}=${String(value)}`);
  return entries.length > 0 ? entries.join(" · ") : null;
}

export function ObservabilityView() {
  const [statusFilter, setStatusFilter] = useState<"" | AgentRunStatus>("");
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<AgentRun | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [eventsError, setEventsError] = useState<string | null>(null);

  const refresh = useCallback(
    async (status: "" | AgentRunStatus, keepSelection: boolean) => {
      try {
        const response = await listAgentRuns(
          status === "" ? {} : { status },
        );
        setRuns(response.runs);
        setLoadError(null);
        if (!keepSelection) {
          setSelectedRun(null);
          setEvents([]);
        }
      } catch (caught) {
        setLoadError(
          caught instanceof Error ? caught.message : "Failed to load runs.",
        );
      }
    },
    [],
  );

  useEffect(() => {
    let alive = true;
    void (async () => {
      await refresh(statusFilter, false);
      if (alive) {
        setInitialLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [statusFilter, refresh]);

  const handleManualRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await refresh(statusFilter, true);
    } finally {
      setRefreshing(false);
    }
  }, [statusFilter, refresh]);

  const handleSelectRun = useCallback(async (run: AgentRun) => {
    setSelectedRun(run);
    setEvents([]);
    setEventsError(null);
    setEventsLoading(true);
    try {
      const response = await listAgentRunEvents(run.run_id);
      setEvents(response.events);
    } catch (caught) {
      setEventsError(
        caught instanceof Error ? caught.message : "Failed to load events.",
      );
    } finally {
      setEventsLoading(false);
    }
  }, []);

  return (
    <div className="mx-auto w-full max-w-6xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Observability"
        description="Every agent run traced end-to-end in PostgreSQL — LLM calls, tool executions, memory, RAG, and approval decisions — tied together by a stable run id returned with each reply."
        actions={
          <button
            type="button"
            onClick={() => void handleManualRefresh()}
            disabled={refreshing || initialLoading}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <IconRefresh
              className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`}
            />
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        }
      />

      {/* How this works */}
      <div className="mt-6 flex items-start gap-3 rounded-xl border border-indigo-100 bg-indigo-50/60 px-4 py-3.5 text-sm text-indigo-900">
        <IconPulse className="mt-0.5 h-4.5 w-4.5 shrink-0 text-indigo-500" />
        <p className="leading-relaxed">
          Records are safe by construction: compact metadata only — no
          prompts, no raw responses, no secrets, no stack traces. Tracing
          failures are swallowed so they can never break a business
          request.
        </p>
      </div>

      {/* Status filter */}
      <div className="mt-6 flex flex-wrap items-center gap-2">
        {STATUS_FILTERS.map((filter) => (
          <button
            key={filter.label}
            type="button"
            onClick={() => setStatusFilter(filter.value)}
            aria-pressed={statusFilter === filter.value}
            className={`rounded-full border px-3.5 py-1.5 text-xs font-semibold transition-colors ${
              statusFilter === filter.value
                ? "border-indigo-600 bg-indigo-600 text-white"
                : "border-slate-300 bg-white text-slate-600 hover:border-indigo-300 hover:text-indigo-700"
            }`}
          >
            {filter.label}
          </button>
        ))}
      </div>

      {/* Initial loading */}
      {initialLoading && (
        <div className="mt-6 space-y-3" aria-label="Loading runs">
          {[0, 1, 2].map((n) => (
            <div
              key={n}
              className="h-16 animate-pulse rounded-xl border border-slate-200 bg-white"
            />
          ))}
        </div>
      )}

      {/* Load error */}
      {!initialLoading && loadError && (
        <div className="mt-6" role="alert">
          <EmptyState
            icon={<IconAlert className="h-6 w-6" />}
            title="Could not load runs"
            description={loadError}
          >
            <button
              type="button"
              onClick={() => void handleManualRefresh()}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500"
            >
              <IconRefresh className="h-4 w-4" />
              Try again
            </button>
          </EmptyState>
        </div>
      )}

      {/* Empty */}
      {!initialLoading && !loadError && runs.length === 0 && (
        <div className="mt-6">
          <EmptyState
            icon={<IconPulse className="h-6 w-6" />}
            title="No traced runs"
            description="Nothing matches this filter yet. Send a message in Chat (or the WhatsApp webhook) and the run will appear here."
          />
        </div>
      )}

      {/* Run list + detail */}
      {!initialLoading && !loadError && runs.length > 0 && (
        <div className="mt-6 grid gap-4 lg:grid-cols-5">
          {/* Master */}
          <div className="space-y-3 lg:col-span-2">
            <p className="text-xs font-medium text-slate-500">
              {runs.length} run{runs.length === 1 ? "" : "s"}, newest first
            </p>
            {runs.map((run) => (
              <button
                key={run.run_id}
                type="button"
                onClick={() => void handleSelectRun(run)}
                aria-pressed={selectedRun?.run_id === run.run_id}
                className={`w-full rounded-xl border px-4 py-3.5 text-left shadow-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 ${
                  selectedRun?.run_id === run.run_id
                    ? "border-indigo-400 bg-indigo-50/60"
                    : "border-slate-200 bg-white hover:border-indigo-200"
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                      STATUS_STYLES[run.status] ??
                      "border-slate-200 bg-slate-50 text-slate-700"
                    }`}
                  >
                    {run.status}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-600">
                    {run.source}
                  </span>
                  <span className="ml-auto font-mono text-[11px] text-slate-400">
                    {run.event_count} evt · {formatDuration(run.duration_ms)}
                  </span>
                </div>
                <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-slate-800">
                  {run.request_preview ?? (
                    <span className="text-slate-400">(no preview)</span>
                  )}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {formatBackendTimestamp(run.started_at)}
                </p>
              </button>
            ))}
          </div>

          {/* Detail */}
          <div className="lg:col-span-3">
            {!selectedRun && (
              <div className="flex h-full min-h-48 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50/60">
                <p className="text-sm text-slate-500">
                  Select a run to inspect its event timeline.
                </p>
              </div>
            )}
            {selectedRun && (
              <article className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                      STATUS_STYLES[selectedRun.status] ??
                      "border-slate-200 bg-slate-50 text-slate-700"
                    }`}
                  >
                    {selectedRun.status}
                  </span>
                  <span className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] font-medium text-slate-600">
                    {selectedRun.source}
                  </span>
                  {selectedRun.owner_key && (
                    <span className="font-mono text-[11px] text-slate-400">
                      owner {selectedRun.owner_key}
                    </span>
                  )}
                  {selectedRun.error_type && (
                    <span className="rounded-full border border-rose-200 bg-rose-50 px-2 py-0.5 text-[11px] font-semibold text-rose-700">
                      {selectedRun.error_type}
                    </span>
                  )}
                </div>
                <p className="mt-2 font-mono text-xs text-slate-500">
                  {selectedRun.run_id}
                </p>
                <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm sm:grid-cols-3">
                  <div>
                    <dt className="text-xs text-slate-500">Started</dt>
                    <dd className="text-slate-800">
                      {formatBackendTimestamp(selectedRun.started_at)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-500">Completed</dt>
                    <dd className="text-slate-800">
                      {formatBackendTimestamp(selectedRun.completed_at)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-slate-500">Duration</dt>
                    <dd className="text-slate-800">
                      {formatDuration(selectedRun.duration_ms)}
                    </dd>
                  </div>
                </dl>

                <h3 className="mt-5 text-sm font-semibold text-slate-900">
                  Event timeline
                </h3>
                {eventsLoading && (
                  <div
                    className="mt-3 space-y-2"
                    aria-label="Loading events"
                  >
                    {[0, 1, 2].map((n) => (
                      <div
                        key={n}
                        className="h-12 animate-pulse rounded-lg border border-slate-200 bg-slate-50"
                      />
                    ))}
                  </div>
                )}
                {eventsError && (
                  <p
                    role="alert"
                    className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3.5 py-2.5 text-sm text-rose-700"
                  >
                    {eventsError}
                  </p>
                )}
                {!eventsLoading && !eventsError && events.length === 0 && (
                  <p className="mt-3 text-sm text-slate-500">
                    No events recorded for this run.
                  </p>
                )}
                {!eventsLoading && !eventsError && events.length > 0 && (
                  <ol className="mt-3 space-y-2">
                    {events.map((event) => (
                      <li
                        key={event.event_id}
                        className="rounded-lg border border-slate-200 bg-slate-50/60 px-3.5 py-2.5"
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span
                            className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                              EVENT_TYPE_STYLES[event.event_type] ??
                              "border-slate-200 bg-slate-50 text-slate-700"
                            }`}
                          >
                            {event.event_type}
                          </span>
                          <span className="text-sm font-medium text-slate-800">
                            {eventSummary(event)}
                          </span>
                          <span
                            className={`text-[11px] font-semibold ${
                              event.status === "failed"
                                ? "text-rose-600"
                                : "text-slate-500"
                            }`}
                          >
                            {event.status}
                          </span>
                          <span className="ml-auto font-mono text-[11px] text-slate-400">
                            {event.iteration !== null && `it${event.iteration} · `}
                            {formatDuration(event.duration_ms)}
                          </span>
                        </div>
                        {event.error_type && (
                          <p className="mt-1 text-xs font-medium text-rose-600">
                            {event.error_type}
                          </p>
                        )}
                        {eventMetaLine(event) && (
                          <p className="mt-1 truncate font-mono text-[11px] text-slate-500">
                            {eventMetaLine(event)}
                          </p>
                        )}
                      </li>
                    ))}
                  </ol>
                )}
              </article>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
