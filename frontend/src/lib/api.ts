/**
 * Centralized client for the DibantuAI FastAPI backend.
 *
 * Every network call the frontend makes goes through this module. The
 * backend base URL comes from NEXT_PUBLIC_API_URL (see
 * .env.local.example); nothing else in the codebase hardcodes it.
 *
 * Backend endpoints used (and nothing else — no endpoint is invented):
 *   GET  /health
 *   POST /api/chat
 *   GET  /api/approvals
 *   POST /api/approvals/{id}/approve
 *   POST /api/approvals/{id}/reject
 *   POST /api/approvals/{id}/execute
 *   GET  /api/memory
 *   DELETE /api/memory/{id}
 *   GET  /api/observability/runs
 *   GET  /api/observability/runs/{run_id}/events
 */

/** Base URL of the FastAPI backend, without a trailing slash. */
export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

/** Host:port shown in the UI so developers can see where requests go. */
export const API_HOST_LABEL = (() => {
  try {
    return new URL(API_BASE_URL).host;
  } catch {
    return API_BASE_URL;
  }
})();

/**
 * Error for every non-2xx or unreachable request. FastAPI failures
 * carry a human-readable `detail` string, which we surface as-is.
 */
export class ApiError extends Error {
  /** HTTP status code, or 0 when the backend could not be reached. */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// ---------------------------------------------------------------------------
// Response types (mirror app/models/schemas.py)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface ChatResponse {
  response: string;
}

/** Approval lifecycle statuses (Step 16 adds the execution states). */
export type ApprovalStatus =
  | "pending"
  | "approved"
  | "rejected"
  | "executing"
  | "executed"
  | "failed";

export interface Approval {
  approval_id: string;
  action: string;
  requested_by: string;
  payload: Record<string, unknown>;
  status: ApprovalStatus;
  created_at: string;
  decided_at: string | null;
  execution_started_at: string | null;
  executed_at: string | null;
  /** Business-level failure reason (bounded; never a traceback). */
  execution_error: string | null;
  /** The sensitive tool's result dict on successful execution. */
  execution_result: Record<string, unknown> | null;
}

export interface ApprovalListResponse {
  approvals: Approval[];
  count: number;
}

/** Memory types offered by the backend (Step 14). */
export type MemoryType =
  | "preference"
  | "customer_context"
  | "business_context"
  | "instruction";

export interface Memory {
  memory_id: number;
  owner_key: string;
  memory_type: MemoryType;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface MemoryListResponse {
  memories: Memory[];
  count: number;
}

/** Run statuses reported by the observability store (Step 15). */
export type AgentRunStatus = "running" | "completed" | "failed";

export interface AgentRun {
  run_id: string;
  source: string;
  owner_key: string | null;
  status: AgentRunStatus;
  request_preview: string | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  error_type: string | null;
  event_count: number;
}

export interface AgentRunListResponse {
  runs: AgentRun[];
  count: number;
}

/** Event types traced inside a run (Step 15). */
export type AgentEventType =
  | "RUN"
  | "LLM"
  | "TOOL"
  | "MEMORY"
  | "RAG"
  | "APPROVAL";

export interface AgentEvent {
  event_id: number;
  run_id: string;
  event_type: AgentEventType;
  event_name: string;
  status: string;
  iteration: number | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  metadata: Record<string, unknown> | null;
  error_type: string | null;
}

export interface AgentEventListResponse {
  run_id: string;
  events: AgentEvent[];
  count: number;
}

// ---------------------------------------------------------------------------
// Core request helper
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(
      `Cannot reach the DibantuAI backend at ${API_BASE_URL}. Is the backend running?`,
      0,
    );
  }

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof (body as { detail: unknown }).detail === "string"
      ) {
        message = (body as { detail: string }).detail;
      }
    } catch {
      // Non-JSON error body — keep the generic message.
    }
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// API operations
// ---------------------------------------------------------------------------

/** POST /api/chat — send a user message, get the assistant's reply. */
export function sendMessage(message: string): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}

/** GET /health — liveness probe, also used for the connection badge. */
export function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/** Statuses GET /api/approvals?status= accepts ("all" = every approval). */
export type ApprovalStatusFilter = ApprovalStatus | "all";

/**
 * GET /api/approvals — sensitive-action approvals, oldest first.
 * Without a filter: pending only (the original behavior). "all" returns
 * the full lifecycle so the Approvals page can offer execution.
 */
export function listApprovals(
  status: ApprovalStatusFilter = "pending",
): Promise<ApprovalListResponse> {
  const suffix = status === "pending" ? "" : `?status=${status}`;
  return request<ApprovalListResponse>(`/api/approvals${suffix}`);
}

/** POST /api/approvals/{id}/approve — approve a pending approval. */
export function approveApproval(approvalId: string): Promise<Approval> {
  return request<Approval>(
    `/api/approvals/${encodeURIComponent(approvalId)}/approve`,
    { method: "POST" },
  );
}

/** POST /api/approvals/{id}/reject — reject a pending approval. */
export function rejectApproval(approvalId: string): Promise<Approval> {
  return request<Approval>(
    `/api/approvals/${encodeURIComponent(approvalId)}/reject`,
    { method: "POST" },
  );
}

/**
 * POST /api/approvals/{id}/execute — run an APPROVED action exactly once
 * (Step 16). Sends NO body on purpose: the backend executes only the
 * immutable payload snapshot it stored when the approval was created;
 * the client can never re-supply or modify it.
 */
export function executeApproval(approvalId: string): Promise<Approval> {
  return request<Approval>(
    `/api/approvals/${encodeURIComponent(approvalId)}/execute`,
    { method: "POST" },
  );
}

/**
 * GET /api/memory — one owner's memories, newest first. `ownerKey` is
 * application-level ownership (the backend has no authentication yet).
 */
export function listMemories(
  ownerKey: string = "default",
): Promise<MemoryListResponse> {
  return request<MemoryListResponse>(
    `/api/memory?owner_key=${encodeURIComponent(ownerKey)}`,
  );
}

/** DELETE /api/memory/{id} — forget one of the owner's memories. */
export function deleteMemory(
  memoryId: number,
  ownerKey: string = "default",
): Promise<Memory> {
  return request<Memory>(
    `/api/memory/${memoryId}?owner_key=${encodeURIComponent(ownerKey)}`,
    { method: "DELETE" },
  );
}

/** Optional filters for GET /api/observability/runs (Step 15). */
export interface AgentRunFilters {
  status?: AgentRunStatus;
  source?: string;
  ownerKey?: string;
}

/** GET /api/observability/runs — recent traced runs, newest first. */
export function listAgentRuns(
  filters: AgentRunFilters = {},
): Promise<AgentRunListResponse> {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.source) params.set("source", filters.source);
  if (filters.ownerKey) params.set("owner_key", filters.ownerKey);
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return request<AgentRunListResponse>(`/api/observability/runs${suffix}`);
}

/**
 * GET /api/observability/runs/{run_id}/events — one run's event
 * timeline in occurrence order. Run details come from the list call.
 */
export function listAgentRunEvents(
  runId: string,
  eventType?: AgentEventType,
): Promise<AgentEventListResponse> {
  const suffix = eventType ? `?event_type=${eventType}` : "";
  return request<AgentEventListResponse>(
    `/api/observability/runs/${encodeURIComponent(runId)}/events${suffix}`,
  );
}
