"use client";

/**
 * Orders page. The table is fed by the real backend through
 * GET /api/orders (read-only, straight from PostgreSQL — the same
 * source of truth the create_order tool writes), while the assistant
 * panel above it stays for actions ("buat order", refunds, customer
 * lookup). Loading, empty, and error states are all first-class; a
 * page refresh always re-fetches from the backend.
 */

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AssistantAsk } from "@/components/assistant-ask";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import {
  IconAlert,
  IconArrowRight,
  IconCart,
  IconCheck,
  IconInbox,
  IconRefresh,
} from "@/components/icons";
import { getOrders, type Order } from "@/lib/api";
import { formatBackendTimestamp, formatRupiah } from "@/lib/format";

const FLOW = [
  {
    title: "Create an order",
    body: "Tell the assistant what to order and for whom — it registers the customer and the items in one step.",
  },
  {
    title: "Refunds & cancellations",
    body: "Refunding or cancelling creates a pending approval instead of executing — a human decides on the Approvals page.",
  },
  {
    title: "Customer lookup",
    body: "Ask for a customer by name to see their record and order history.",
  },
];

/** Status chip styles — mirrors the inventory/approval badge language. */
const STATUS_STYLES = {
  completed: {
    label: "Completed",
    style: "border-emerald-200 bg-emerald-50 text-emerald-700",
  },
  refunded: {
    label: "Refunded",
    style: "border-rose-200 bg-rose-50 text-rose-700",
  },
  cancelled: {
    label: "Cancelled",
    style: "border-amber-200 bg-amber-50 text-amber-700",
  },
} as const;

function orderStatus(status: string): { label: string; style: string } {
  const known = STATUS_STYLES[status as keyof typeof STATUS_STYLES];
  return known ?? {
    label: status,
    style: "border-slate-200 bg-slate-50 text-slate-600",
  };
}

const TABLE_COLUMNS = [
  "Order",
  "Customer",
  "Items",
  "Total",
  "Status",
  "Placed",
] as const;
const GRID_STYLE = {
  gridTemplateColumns:
    "minmax(0, 0.8fr) minmax(0, 1.1fr) minmax(0, 1.8fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.2fr)",
} as const;

function OrderItems({ order }: { order: Order }) {
  if (order.items.length === 0) {
    return <span className="text-sm text-slate-400">No items</span>;
  }
  return (
    <div className="flex flex-col gap-0.5">
      {order.items.map((item, index) => (
        <div
          key={`${item.product_name}-${index}`}
          className="truncate text-sm text-slate-700"
        >
          <span className="font-semibold text-slate-900">
            {item.quantity}×
          </span>{" "}
          {item.product_name}
        </div>
      ))}
    </div>
  );
}

export function OrdersView() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await getOrders();
      setOrders(response.orders);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : "Failed to load orders.",
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

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Orders"
        description="Order history straight from the backend database — the same source of truth the assistant writes. Ask the assistant for actions; read the table for the full picture."
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

      <div className="mt-6">
        <AssistantAsk
          placeholder='e.g. Buat order untuk Budi: 2 Kopi Susu'
          suggestions={[
            "Buat order untuk Budi: 2 Kopi Susu",
            "Cari customer Budi",
            "Refund order terakhir",
          ]}
        />
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {FLOW.map((step, index) => (
          <div
            key={step.title}
            className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-xs font-bold text-indigo-600">
              {index + 1}
            </span>
            <h3 className="mt-3 text-sm font-semibold text-slate-900">
              {step.title}
            </h3>
            <p className="mt-1.5 text-sm leading-relaxed text-slate-600">
              {step.body}
            </p>
          </div>
        ))}
      </div>

      <p className="mt-4 text-sm text-slate-500">
        Asking for a refund?{" "}
        <Link
          href="/approvals"
          className="inline-flex items-center gap-1 font-medium text-indigo-600 hover:text-indigo-500 hover:underline"
        >
          Review it on the Approvals page
          <IconArrowRight className="h-3.5 w-3.5" />
        </Link>
      </p>

      <div className="mt-6">
        {/* Loading skeleton — same shape as the real table. */}
        {loading && (
          <section
            aria-label="Loading orders"
            className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
          >
            <div
              className="grid gap-4 border-b border-slate-100 bg-slate-50/60 px-5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400"
              style={GRID_STYLE}
            >
              {TABLE_COLUMNS.map((column) => (
                <span key={column}>{column}</span>
              ))}
            </div>
            {[0, 1, 2, 3].map((row) => (
              <div
                key={row}
                className="grid gap-4 border-b border-slate-50 px-5 py-3.5 last:border-b-0"
                style={GRID_STYLE}
              >
                {TABLE_COLUMNS.map((column) => (
                  <span
                    key={column}
                    className="h-3 rounded bg-slate-100"
                    aria-hidden="true"
                  />
                ))}
              </div>
            ))}
          </section>
        )}

        {/* Load error */}
        {!loading && loadError && (
          <div role="alert">
            <EmptyState
              icon={<IconAlert className="h-6 w-6" />}
              title="Could not load orders"
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
        {!loading && !loadError && orders.length === 0 && (
          <EmptyState
            icon={<IconInbox className="h-6 w-6" />}
            title="No orders yet"
            description="Orders appear here as soon as they are created through the assistant or the chat."
          />
        )}

        {/* Loaded: the real table, straight from PostgreSQL. */}
        {!loading && !loadError && orders.length > 0 && (
          <section
            aria-label="Order history table"
            className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
          >
            <header className="flex flex-col gap-1 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-sm font-semibold text-slate-900">
                Order history
              </h2>
              <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700">
                <IconCheck className="h-3 w-3" aria-hidden="true" />
                {orders.length} orders · GET /api/orders
              </span>
            </header>
            <div
              className="grid gap-4 border-b border-slate-100 bg-slate-50/60 px-5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400"
              style={GRID_STYLE}
            >
              {TABLE_COLUMNS.map((column) => (
                <span key={column}>{column}</span>
              ))}
            </div>
            {orders.map((order) => {
              const status = orderStatus(order.status);
              return (
                <div
                  key={order.order_id}
                  className="grid items-center gap-4 border-b border-slate-50 px-5 py-3.5 last:border-b-0"
                  style={GRID_STYLE}
                >
                  <span className="font-mono text-xs font-semibold text-slate-900">
                    {order.order_id}
                  </span>
                  <span className="min-w-0 truncate text-sm font-medium text-slate-900">
                    {order.customer?.name ?? "—"}
                  </span>
                  <OrderItems order={order} />
                  <span className="font-mono text-xs text-slate-700">
                    {formatRupiah(order.total_price)}
                  </span>
                  <span>
                    <span
                      className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${status.style}`}
                    >
                      {status.label}
                    </span>
                  </span>
                  <span className="text-xs text-slate-500">
                    {formatBackendTimestamp(order.created_at)}
                  </span>
                </div>
              );
            })}
          </section>
        )}
      </div>

      <div className="mt-6 flex items-start gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3.5 text-sm text-slate-600 shadow-sm">
        <IconCart className="mt-0.5 h-4.5 w-4.5 shrink-0 text-slate-400" />
        <p className="leading-relaxed">
          Prefer the conversational route? The{" "}
          <Link
            href="/chat"
            className="font-medium text-indigo-600 hover:text-indigo-500 hover:underline"
          >
            full chat experience
          </Link>{" "}
          supports multi-step ordering with the same tools.
        </p>
      </div>
    </div>
  );
}
