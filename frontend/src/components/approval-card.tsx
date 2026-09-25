"use client";

/**
 * Card for one sensitive-action approval across its whole lifecycle
 * (GET /api/approvals?status=all): pending cards offer the human
 * decision, approved cards offer the explicit Execute action (Step 16),
 * and terminal cards show the outcome — result, error, or rejection.
 */

import { IconCheck, IconPulse, IconShield, IconX } from "@/components/icons";
import type { Approval } from "@/lib/api";
import { formatBackendTimestamp, humanizeAction } from "@/lib/format";

const ACTION_STYLES: Record<string, string> = {
  refund_order: "border-amber-200 bg-amber-50 text-amber-700",
  cancel_order: "border-rose-200 bg-rose-50 text-rose-700",
  bulk_stock_update: "border-violet-200 bg-violet-50 text-violet-700",
};

const STATUS_STYLES: Record<string, string> = {
  pending: "border-amber-200 bg-amber-50 text-amber-700",
  approved: "border-emerald-200 bg-emerald-50 text-emerald-700",
  rejected: "border-slate-300 bg-slate-100 text-slate-600",
  executing: "border-indigo-200 bg-indigo-50 text-indigo-700",
  executed: "border-sky-200 bg-sky-50 text-sky-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
};

function payloadValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

export function ApprovalCard({
  approval,
  deciding,
  executing = false,
  onDecide,
  onExecute,
}: {
  approval: Approval;
  /** Which decision is in flight for this card, if any. */
  deciding: "approve" | "reject" | null;
  /** Whether the Execute request for this card is in flight. */
  executing?: boolean;
  onDecide: (approval: Approval, decision: "approve" | "reject") => void;
  onExecute: (approval: Approval) => void;
}) {
  const busy = deciding !== null || executing;
  const actionStyle = ACTION_STYLES[approval.action] ?? "border-slate-200 bg-slate-50 text-slate-700";
  const statusStyle = STATUS_STYLES[approval.status] ?? "border-slate-200 bg-slate-50 text-slate-700";
  const payloadEntries = Object.entries(approval.payload);
  // Scalar result fields only — the safe, compact outcome view.
  const resultEntries = Object.entries(approval.execution_result ?? {}).filter(
    ([, value]) =>
      value === null ||
      typeof value !== "object",
  );

  return (
    <article
      aria-labelledby={`approval-${approval.approval_id}`}
      className="rounded-xl border border-slate-200 bg-white shadow-sm transition-opacity disabled:opacity-60"
    >
      <div className="flex flex-col gap-4 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-slate-500">
            <IconShield className="h-4.5 w-4.5" />
          </span>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3
                id={`approval-${approval.approval_id}`}
                className="text-sm font-semibold text-slate-900"
              >
                {humanizeAction(approval.action)}
              </h3>
              <span
                className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${actionStyle}`}
              >
                {approval.action}
              </span>
              <span
                className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusStyle}`}
              >
                {approval.status === "executing" ? (
                  <span className="inline-flex items-center gap-1">
                    <IconPulse className="h-3 w-3 animate-pulse" />
                    executing
                  </span>
                ) : (
                  approval.status
                )}
              </span>
            </div>
            <p className="mt-1 font-mono text-xs text-slate-500">
              {approval.approval_id}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Requested by <span className="font-medium text-slate-700">{approval.requested_by}</span>{" "}
              · {formatBackendTimestamp(approval.created_at)}
            </p>
          </div>
        </div>

        {/* Lifecycle actions: decide while pending, execute once approved. */}
        <div className="flex shrink-0 items-center gap-2">
          {approval.status === "pending" && (
            <>
              <button
                type="button"
                disabled={busy}
                onClick={() => onDecide(approval, "approve")}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3.5 py-2 text-sm font-semibold text-white transition-colors hover:bg-emerald-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <IconCheck className="h-4 w-4" />
                {deciding === "approve" ? "Approving…" : "Approve"}
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => onDecide(approval, "reject")}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-semibold text-slate-700 transition-colors hover:border-rose-300 hover:bg-rose-50 hover:text-rose-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <IconX className="h-4 w-4" />
                {deciding === "reject" ? "Rejecting…" : "Reject"}
              </button>
            </>
          )}
          {approval.status === "approved" && (
            <button
              type="button"
              disabled={busy}
              onClick={() => onExecute(approval)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <IconPulse className={`h-4 w-4 ${executing ? "animate-pulse" : ""}`} />
              {executing ? "Executing…" : "Execute"}
            </button>
          )}
          {approval.status === "executing" && (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-200 bg-indigo-50 px-3.5 py-2 text-sm font-medium text-indigo-700">
              <IconPulse className="h-4 w-4 animate-pulse" />
              Executing…
            </span>
          )}
          {(approval.status === "executed" ||
            approval.status === "failed" ||
            approval.status === "rejected") && (
            <span className="text-xs text-slate-400">no actions left</span>
          )}
        </div>
      </div>

      {/* Payload */}
      <div className="px-5 py-4">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
          Action details
        </p>
        {payloadEntries.length > 0 ? (
          <dl className="mt-2 grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {payloadEntries.map(([key, value]) => (
              <div
                key={key}
                className="flex min-w-0 items-baseline justify-between gap-3 border-b border-dashed border-slate-100 py-1"
              >
                <dt className="shrink-0 text-xs font-medium text-slate-500">
                  {key.replace(/_/g, " ")}
                </dt>
                <dd className="min-w-0 truncate font-mono text-xs text-slate-800">
                  {payloadValue(value)}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="mt-2 text-xs text-slate-400">
            No additional details were provided with this request.
          </p>
        )}
      </div>

      {/* Execution outcome (Step 16) */}
      {approval.status === "executed" && resultEntries.length > 0 && (
        <div className="border-t border-slate-100 px-5 py-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-sky-600">
            Execution result
          </p>
          <dl className="mt-2 grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {resultEntries.map(([key, value]) => (
              <div
                key={key}
                className="flex min-w-0 items-baseline justify-between gap-3 border-b border-dashed border-slate-100 py-1"
              >
                <dt className="shrink-0 text-xs font-medium text-slate-500">
                  {key.replace(/_/g, " ")}
                </dt>
                <dd className="min-w-0 truncate font-mono text-xs text-slate-800">
                  {payloadValue(value)}
                </dd>
              </div>
            ))}
          </dl>
          {approval.executed_at && (
            <p className="mt-2 text-xs text-slate-500">
              Executed {formatBackendTimestamp(approval.executed_at)}
            </p>
          )}
        </div>
      )}

      {approval.status === "failed" && (
        <div className="border-t border-slate-100 px-5 py-4" role="alert">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-rose-600">
            Execution failed
          </p>
          <p className="mt-2 text-sm leading-relaxed text-rose-700">
            {approval.execution_error ?? "The action could not be executed."}
          </p>
          {approval.executed_at && (
            <p className="mt-2 text-xs text-slate-500">
              Attempted {formatBackendTimestamp(approval.executed_at)}
            </p>
          )}
        </div>
      )}
    </article>
  );
}
