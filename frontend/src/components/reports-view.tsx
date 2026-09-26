"use client";

/**
 * Reports page. The summary cards are fed by the real backend through
 * GET /api/reports (read-only, straight from PostgreSQL — the same
 * sales_window aggregation the get_sales_report tool runs), while the
 * assistant cards above stay for on-demand narrative reports. Loading,
 * empty, and error states are all first-class; a page refresh always
 * re-fetches from the backend.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  AssistantAsk,
  type AssistantAskHandle,
} from "@/components/assistant-ask";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import {
  IconAlert,
  IconChart,
  IconCheck,
  IconInbox,
  IconRefresh,
  IconSparkles,
} from "@/components/icons";
import { getReports, type ReportsResponse } from "@/lib/api";
import { formatBackendTimestamp, formatRupiah } from "@/lib/format";

const REPORTS = [
  {
    id: "daily",
    label: "Daily",
    description: "Today's sales performance at a glance",
    prompt: "Buat laporan penjualan harian",
  },
  {
    id: "weekly",
    label: "Weekly",
    description: "The last seven days, summarized",
    prompt: "Buat laporan penjualan mingguan",
  },
  {
    id: "monthly",
    label: "Monthly",
    description: "Where the month stands so far",
    prompt: "Buat laporan penjualan bulanan",
  },
] as const;

const PERIOD_LABELS: Record<string, string> = {
  daily: "Daily",
  weekly: "Weekly",
  monthly: "Monthly",
};

/** Status chip styles — mirrors the orders/approval badge language. */
const STATUS_STYLES = {
  completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
  refunded: "border-rose-200 bg-rose-50 text-rose-700",
  cancelled: "border-amber-200 bg-amber-50 text-amber-700",
} as const;

const KNOWN_STATUSES = Object.keys(STATUS_STYLES);

function statusStyle(status: string): string {
  return (
    STATUS_STYLES[status as keyof typeof STATUS_STYLES] ??
    "border-slate-200 bg-slate-50 text-slate-600"
  );
}

/** Known statuses first (fixed order), then any other stored status. */
function statusEntries(summary: ReportsResponse["status_summary"]) {
  const byStatus = summary.by_status;
  const known = KNOWN_STATUSES.filter((status) => status in byStatus);
  const extra = Object.keys(byStatus).filter(
    (status) => !KNOWN_STATUSES.includes(status),
  );
  return [...known, ...extra].map(
    (status) => [status, byStatus[status]] as const,
  );
}

export function ReportsView() {
  const askRef = useRef<AssistantAskHandle>(null);
  const [report, setReport] = useState<ReportsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await getReports();
      setReport(response);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : "Failed to load reports.",
      );
    }
  }, []);

  useEffect(() => {
    let alive = true;
    void (async () => {
      await refresh();
      if (alive) {
        setLoading(false);
      }
    })();
    return () => {
      alive = false;
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

  const hasSales = (report?.status_summary.total ?? 0) > 0;

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Reports"
        description="Sales summaries straight from the backend database — the same source of truth the assistant reads. Ask the assistant for a narrative report; read the cards for the full picture."
        actions={
          <button
            type="button"
            onClick={() => void handleManualRefresh()}
            disabled={refreshing || loading}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <IconRefresh
              className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`}
            />
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        }
      />

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {REPORTS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => askRef.current?.ask(preset.prompt)}
            className="group rounded-xl border border-slate-200 bg-white p-5 text-left shadow-sm transition-colors hover:border-indigo-300 hover:bg-indigo-50/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-500 transition-colors group-hover:bg-indigo-100 group-hover:text-indigo-600">
              <IconChart className="h-4.5 w-4.5" />
            </span>
            <h3 className="mt-3 text-sm font-semibold text-slate-900">
              {preset.label} sales
            </h3>
            <p className="mt-1 text-sm leading-relaxed text-slate-500">
              {preset.description}
            </p>
            <span className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-600">
              <IconSparkles className="h-3.5 w-3.5" />
              Generate via assistant
            </span>
          </button>
        ))}
      </div>

      <div className="mt-6">
        <AssistantAsk
          ref={askRef}
          placeholder="e.g. Buat laporan penjualan harian"
          suggestions={["Ringkas penjualan hari ini", "Produk terlaris hari ini"]}
        />
      </div>

      <div className="mt-6">
        {/* Loading skeleton — same shape as the real cards. */}
        {loading && (
          <section
            aria-label="Loading sales summary"
            className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
          >
            <div className="flex flex-col gap-1 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <span className="h-3 w-32 rounded bg-slate-100" aria-hidden="true" />
              <span className="h-3 w-40 rounded bg-slate-100" aria-hidden="true" />
            </div>
            <div className="grid gap-4 px-5 py-4 sm:grid-cols-3">
              {[0, 1, 2].map((card) => (
                <div key={card} className="rounded-xl border border-slate-100 p-4">
                  <span
                    className="block h-3 w-16 rounded bg-slate-100"
                    aria-hidden="true"
                  />
                  <span
                    className="mt-3 block h-6 w-24 rounded bg-slate-100"
                    aria-hidden="true"
                  />
                  <span
                    className="mt-2 block h-3 w-28 rounded bg-slate-100"
                    aria-hidden="true"
                  />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Load error */}
        {!loading && loadError && (
          <div role="alert">
            <EmptyState
              icon={<IconAlert className="h-6 w-6" />}
              title="Could not load reports"
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

        {/* Empty order history */}
        {!loading && !loadError && report && !hasSales && (
          <EmptyState
            icon={<IconInbox className="h-6 w-6" />}
            title="No sales data yet"
            description="Sales figures appear here as soon as the first order exists in the backend database."
          />
        )}

        {/* Loaded: real aggregates, straight from PostgreSQL. */}
        {!loading && !loadError && report && hasSales && (
          <section
            aria-label="Sales summary"
            className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
          >
            <header className="flex flex-col gap-1 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-sm font-semibold text-slate-900">
                Sales summary
              </h2>
              <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700">
                <IconCheck className="h-3 w-3" aria-hidden="true" />
                {report.status_summary.total} orders · GET /api/reports
              </span>
            </header>

            <div className="flex flex-wrap items-center gap-2 border-b border-slate-100 px-5 py-3.5">
              {statusEntries(report.status_summary).map(([status, count]) => (
                <span
                  key={status}
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold ${statusStyle(status)}`}
                >
                  {status}
                  <span className="font-mono">{count}</span>
                </span>
              ))}
              <span className="ml-auto text-[11px] text-slate-400">
                as of {formatBackendTimestamp(report.generated_at)}
              </span>
            </div>

            <div className="grid gap-4 px-5 py-4 sm:grid-cols-3">
              {report.windows.map((window) => (
                <div
                  key={window.period}
                  className="rounded-xl border border-slate-100 p-4"
                >
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    {PERIOD_LABELS[window.period] ?? window.period}
                  </h3>
                  <p className="mt-2 font-mono text-lg font-semibold text-slate-900">
                    {formatRupiah(window.total_revenue)}
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    {window.total_orders} orders ·{" "}
                    {window.total_items_sold} items
                  </p>
                  <p className="mt-2 truncate text-xs text-slate-600">
                    Top:{" "}
                    <span className="font-medium text-slate-900">
                      {window.top_product ?? "—"}
                    </span>
                  </p>
                </div>
              ))}
            </div>

            <p className="border-t border-slate-100 bg-slate-50/60 px-5 py-3 text-[11px] leading-relaxed text-slate-500">
              Windows cover completed orders only — refunded and cancelled
              orders are excluded because their stock was restored and their
              revenue undone. Figures match the assistant&apos;s
              get_sales_report answers (same database aggregation).
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
