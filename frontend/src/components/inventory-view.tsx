"use client";

/**
 * Inventory page. The table is fed by the real backend through
 * GET /api/inventory (read-only, straight from PostgreSQL — the same
 * source of truth the check_stock tool reads), while the assistant
 * panel next to it stays for actions ("cek stok", restock, etc.).
 * Loading, empty, and error states are all first-class; a page
 * refresh always re-fetches from the backend.
 */

import { useCallback, useEffect, useState } from "react";

import { AssistantAsk } from "@/components/assistant-ask";
import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import {
  IconAlert,
  IconBoxes,
  IconCheck,
  IconInbox,
  IconRefresh,
} from "@/components/icons";
import { getInventory, type InventoryProduct } from "@/lib/api";
import { formatRupiah } from "@/lib/format";

const CAPABILITIES = [
  "Check any product's current stock",
  "See which products are running low",
  "Adjust stock levels (bulk updates need approval)",
];

/** Status chip styles — mirrors the approval-card badge language. */
const STATUS_STYLES = {
  out: "border-rose-200 bg-rose-50 text-rose-700",
  low: "border-amber-200 bg-amber-50 text-amber-700",
  ok: "border-emerald-200 bg-emerald-50 text-emerald-700",
} as const;

function stockStatus(product: InventoryProduct): {
  label: string;
  style: string;
} {
  if (!product.in_stock) {
    return { label: "Out of stock", style: STATUS_STYLES.out };
  }
  if (product.low_stock) {
    return { label: "Low stock", style: STATUS_STYLES.low };
  }
  return { label: "In stock", style: STATUS_STYLES.ok };
}

const TABLE_COLUMNS = ["Product", "Price", "In stock", "Status"] as const;
const GRID_STYLE = {
  gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)",
} as const;

export function InventoryView() {
  const [products, setProducts] = useState<InventoryProduct[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await getInventory();
      setProducts(response.products);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : "Failed to load inventory.",
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
        title="Inventory"
        description="Live stock levels straight from the backend database — the same source of truth the assistant reads. Ask the assistant for actions; read the table for the full picture."
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

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_280px]">
        <AssistantAsk
          placeholder="e.g. Cek stok kopi susu"
          suggestions={[
            "Cek stok kopi susu",
            "Produk apa saja yang stoknya rendah?",
            "Cek stok Teh Tarik",
          ]}
        />

        <aside className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <IconBoxes className="h-4 w-4" />
            </span>
            <h2 className="text-sm font-semibold text-slate-900">
              What you can ask
            </h2>
          </div>
          <ul className="mt-4 space-y-2.5">
            {CAPABILITIES.map((capability) => (
              <li
                key={capability}
                className="flex items-start gap-2 text-sm leading-relaxed text-slate-600"
              >
                <span
                  className="mt-2 h-1 w-1 shrink-0 rounded-full bg-slate-300"
                  aria-hidden="true"
                />
                {capability}
              </li>
            ))}
          </ul>
          <p className="mt-4 border-t border-slate-100 pt-3 text-xs leading-relaxed text-slate-400">
            Sensitive bulk changes are never applied directly — they become
            approvals on the Approvals page.
          </p>
        </aside>
      </div>

      <div className="mt-6">
        {/* Loading skeleton — same shape as the real table. */}
        {loading && (
          <section
            aria-label="Loading inventory"
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
              title="Could not load inventory"
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

        {/* Empty catalog */}
        {!loading && !loadError && products.length === 0 && (
          <EmptyState
            icon={<IconInbox className="h-6 w-6" />}
            title="No products"
            description="The catalog is empty. Products appear here as soon as the backend database has them."
          />
        )}

        {/* Loaded: the real table, straight from PostgreSQL. */}
        {!loading && !loadError && products.length > 0 && (
          <section
            aria-label="Live stock table"
            className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
          >
            <header className="flex flex-col gap-1 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-sm font-semibold text-slate-900">
                Live stock table
              </h2>
              <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700">
                <IconCheck className="h-3 w-3" aria-hidden="true" />
                {products.length} products · GET /api/inventory
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
            {products.map((product) => {
              const status = stockStatus(product);
              return (
                <div
                  key={product.name}
                  className="grid items-center gap-4 border-b border-slate-50 px-5 py-3.5 last:border-b-0"
                  style={GRID_STYLE}
                >
                  <span className="min-w-0 truncate text-sm font-medium text-slate-900">
                    {product.name}
                  </span>
                  <span className="font-mono text-xs text-slate-700">
                    {formatRupiah(product.price)}
                  </span>
                  <span className="text-sm font-semibold text-slate-900">
                    {product.stock}
                    <span className="ml-1 text-xs font-normal text-slate-400">
                      pcs
                    </span>
                  </span>
                  <span>
                    <span
                      className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${status.style}`}
                    >
                      {status.label}
                    </span>
                  </span>
                </div>
              );
            })}
          </section>
        )}
      </div>
    </div>
  );
}
