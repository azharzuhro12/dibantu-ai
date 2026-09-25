"use client";

/**
 * Human-in-the-loop approvals across the full lifecycle, wired to the
 * real backend:
 *   GET  /api/approvals?status=all
 *   POST /api/approvals/{id}/approve
 *   POST /api/approvals/{id}/reject
 *   POST /api/approvals/{id}/execute   (Step 16)
 *
 * Pending approvals wait for a human decision; approved ones wait for
 * the explicit Execute action; executed/failed/rejected ones stay
 * visible as history. Duplicate executes surface the backend's 409
 * (already executing / already executed) as a clear notice — the
 * backend's atomic claim, not the UI, guarantees single execution.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApprovalCard } from "@/components/approval-card";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import {
  IconAlert,
  IconCheck,
  IconInbox,
  IconRefresh,
  IconShield,
} from "@/components/icons";
import {
  ApiError,
  approveApproval,
  type Approval,
  executeApproval,
  listApprovals,
  rejectApproval,
} from "@/lib/api";

type Notice = {
  kind: "success" | "warning" | "error";
  text: string;
};

const POLL_INTERVAL_MS = 30_000;

const NOTICE_STYLES: Record<Notice["kind"], string> = {
  success: "border-emerald-200 bg-emerald-50 text-emerald-800",
  warning: "border-amber-200 bg-amber-50 text-amber-800",
  error: "border-rose-200 bg-rose-50 text-rose-700",
};

export function ApprovalsView() {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deciding, setDeciding] = useState<{
    id: string;
    decision: "approve" | "reject";
  } | null>(null);
  const [executingId, setExecutingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showNotice = useCallback((next: Notice) => {
    setNotice(next);
    if (noticeTimer.current) {
      clearTimeout(noticeTimer.current);
    }
    noticeTimer.current = setTimeout(() => setNotice(null), 8_000);
  }, []);

  useEffect(() => {
    return () => {
      if (noticeTimer.current) {
        clearTimeout(noticeTimer.current);
      }
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const response = await listApprovals("all");
      setApprovals(response.approvals);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : "Failed to load approvals.",
      );
    }
  }, []);

  useEffect(() => {
    let alive = true;

    // Initial load runs as a scheduled callback, then on an interval.
    const first = setTimeout(() => {
      void (async () => {
        await refresh();
        if (alive) {
          setInitialLoading(false);
        }
      })();
    }, 0);
    const interval = setInterval(() => {
      if (!document.hidden) {
        void refresh();
      }
    }, POLL_INTERVAL_MS);

    return () => {
      alive = false;
      clearTimeout(first);
      clearInterval(interval);
    };
  }, [refresh]);

  const handleManualRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await refresh();
    } finally {
      setRefreshing(false);
    }
  }, [refresh]);

  const decide = useCallback(
    async (approval: Approval, decision: "approve" | "reject") => {
      setDeciding({ id: approval.approval_id, decision });
      try {
        if (decision === "approve") {
          await approveApproval(approval.approval_id);
        } else {
          await rejectApproval(approval.approval_id);
        }
        showNotice({
          kind: "success",
          text: `${approval.approval_id} (${approval.action}) was ${decision === "approve" ? "approved — it now waits for an explicit Execute" : "rejected"}.`,
        });
        await refresh();
      } catch (caught) {
        if (caught instanceof ApiError && caught.status === 404) {
          // Someone else decided or removed it — refresh the truth.
          await refresh();
          showNotice({
            kind: "warning",
            text: `${approval.approval_id} was not found on the backend; the list has been refreshed.`,
          });
        } else if (caught instanceof ApiError && caught.status === 409) {
          showNotice({
            kind: "warning",
            text: `${approval.approval_id} was already decided by someone else: ${caught.message}`,
          });
          await refresh();
        } else {
          showNotice({
            kind: "error",
            text:
              caught instanceof Error
                ? caught.message
                : "The decision could not be saved.",
          });
        }
      } finally {
        setDeciding(null);
      }
    },
    [refresh, showNotice],
  );

  const execute = useCallback(
    async (approval: Approval) => {
      setExecutingId(approval.approval_id);
      try {
        const finished = await executeApproval(approval.approval_id);
        if (finished.status === "executed") {
          showNotice({
            kind: "success",
            text: `${approval.approval_id} (${approval.action}) was executed.`,
          });
        } else {
          // The request succeeded; the action itself was refused.
          showNotice({
            kind: "error",
            text: `${approval.approval_id} (${approval.action}) failed: ${finished.execution_error ?? "the action was refused"}`,
          });
        }
        await refresh();
      } catch (caught) {
        if (caught instanceof ApiError && caught.status === 409) {
          showNotice({
            kind: "warning",
            text: `${approval.approval_id} was not executed again: ${caught.message}`,
          });
          await refresh();
        } else if (caught instanceof ApiError && caught.status === 404) {
          await refresh();
          showNotice({
            kind: "warning",
            text: `${approval.approval_id} was not found on the backend; the list has been refreshed.`,
          });
        } else {
          showNotice({
            kind: "error",
            text:
              caught instanceof Error
                ? caught.message
                : "The action could not be executed.",
          });
        }
      } finally {
        setExecutingId(null);
      }
    },
    [refresh, showNotice],
  );

  const pending = approvals.filter((a) => a.status === "pending");
  const awaitingExecution = approvals.filter(
    (a) => a.status === "approved" || a.status === "executing",
  );
  const history = approvals.filter(
    (a) =>
      a.status === "executed" ||
      a.status === "failed" ||
      a.status === "rejected",
  );

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Approvals"
        description="Sensitive actions requested by the assistant — refunds, order cancellations, bulk stock updates — wait here for a human decision, and only then for an explicit Execute."
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
        <IconShield className="mt-0.5 h-4.5 w-4.5 shrink-0 text-indigo-500" />
        <p className="leading-relaxed">
          The AI assistant never executes sensitive actions on its own. When it
          needs one, a request appears here. Approving records your decision
          only — the action runs when you press Execute, exactly once, from
          the details captured at request time.
        </p>
      </div>

      {/* Feedback notices */}
      {notice && (
        <div
          role="status"
          aria-live="polite"
          className={`mt-4 flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${NOTICE_STYLES[notice.kind]}`}
        >
          {notice.kind === "success" ? (
            <IconCheck className="mt-0.5 h-4 w-4 shrink-0" />
          ) : (
            <IconAlert className="mt-0.5 h-4 w-4 shrink-0" />
          )}
          <span className="leading-relaxed">{notice.text}</span>
        </div>
      )}

      {/* Initial loading */}
      {initialLoading && (
        <div className="mt-6 space-y-4" aria-label="Loading approvals">
          {[0, 1].map((n) => (
            <div
              key={n}
              className="h-28 animate-pulse rounded-xl border border-slate-200 bg-white"
            />
          ))}
        </div>
      )}

      {/* Load error */}
      {!initialLoading && loadError && (
        <div className="mt-6" role="alert">
          <EmptyState
            icon={<IconAlert className="h-6 w-6" />}
            title="Could not load approvals"
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

      {!initialLoading && !loadError && approvals.length === 0 && (
        <div className="mt-6">
          <EmptyState
            icon={<IconInbox className="h-6 w-6" />}
            title="No approvals"
            description="Approvals appear here when the assistant requests a sensitive action — for example when you ask it to refund an order, cancel an order, or update stock in bulk."
          />
        </div>
      )}

      {!initialLoading && !loadError && (
        <div className="mt-6 space-y-8">
          {/* Awaiting decision */}
          {pending.length > 0 && (
            <section aria-label="Awaiting decision">
              <p className="text-xs font-medium text-slate-500">
                {pending.length} awaiting decision
              </p>
              <div className="mt-3 space-y-4">
                {pending.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    deciding={
                      deciding?.id === approval.approval_id
                        ? deciding.decision
                        : null
                    }
                    executing={executingId === approval.approval_id}
                    onDecide={(a, decision) => void decide(a, decision)}
                    onExecute={(a) => void execute(a)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Approved — awaiting execution */}
          {awaitingExecution.length > 0 && (
            <section aria-label="Awaiting execution">
              <p className="text-xs font-medium text-slate-500">
                {awaitingExecution.length} approved — awaiting execution
              </p>
              <div className="mt-3 space-y-4">
                {awaitingExecution.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    deciding={null}
                    executing={executingId === approval.approval_id}
                    onDecide={(a, decision) => void decide(a, decision)}
                    onExecute={(a) => void execute(a)}
                  />
                ))}
              </div>
            </section>
          )}

          {/* History */}
          {history.length > 0 && (
            <section aria-label="Execution history">
              <p className="text-xs font-medium text-slate-500">
                {history.length} finished
              </p>
              <div className="mt-3 space-y-4">
                {history.map((approval) => (
                  <ApprovalCard
                    key={approval.approval_id}
                    approval={approval}
                    deciding={null}
                    executing={false}
                    onDecide={(a, decision) => void decide(a, decision)}
                    onExecute={(a) => void execute(a)}
                  />
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
