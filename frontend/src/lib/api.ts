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

export type ApprovalStatus = "pending" | "approved" | "rejected";

export interface Approval {
  approval_id: string;
  action: string;
  requested_by: string;
  payload: Record<string, unknown>;
  status: ApprovalStatus;
  created_at: string;
  decided_at: string | null;
}

export interface ApprovalListResponse {
  approvals: Approval[];
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

/** GET /api/approvals — pending sensitive-action approvals, oldest first. */
export function listApprovals(): Promise<ApprovalListResponse> {
  return request<ApprovalListResponse>("/api/approvals");
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
