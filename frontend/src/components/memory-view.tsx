"use client";

/**
 * Agent memory browser (Step 14), wired to the real backend:
 *   GET    /api/memory?owner_key=...
 *   DELETE /api/memory/{id}?owner_key=...
 *
 * Simple by design: pick an owner scope, list what the agent remembers
 * for them (type, content, timestamps), and delete entries. Owner keys
 * are application-level ownership, not authentication.
 */

import { useCallback, useEffect, useState } from "react";

import { EmptyState } from "@/components/empty-state";
import { PageHeader } from "@/components/page-header";
import { IconAlert, IconRefresh, IconSparkles, IconX } from "@/components/icons";
import { ApiError, deleteMemory, listMemories, type Memory } from "@/lib/api";
import { formatBackendTimestamp } from "@/lib/format";

const TYPE_STYLES: Record<string, string> = {
  preference: "border-emerald-200 bg-emerald-50 text-emerald-700",
  customer_context: "border-sky-200 bg-sky-50 text-sky-700",
  business_context: "border-amber-200 bg-amber-50 text-amber-700",
  instruction: "border-violet-200 bg-violet-50 text-violet-700",
};

export function MemoryView() {
  const [ownerKey, setOwnerKey] = useState("default");
  const [query, setQuery] = useState("default");
  const [memories, setMemories] = useState<Memory[]>([]);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refresh = useCallback(async (owner: string) => {
    try {
      const response = await listMemories(owner);
      setMemories(response.memories);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : "Failed to load memories.",
      );
    }
  }, []);

  useEffect(() => {
    let alive = true;
    void (async () => {
      await refresh(query);
      if (alive) {
        setInitialLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [query, refresh]);

  const handleManualRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await refresh(query);
    } finally {
      setRefreshing(false);
    }
  }, [query, refresh]);

  const handleDelete = useCallback(
    async (memory: Memory) => {
      setDeletingId(memory.memory_id);
      setNotice(null);
      try {
        await deleteMemory(memory.memory_id, query);
        setMemories((current) =>
          current.filter((m) => m.memory_id !== memory.memory_id),
        );
        setNotice(`Memory ${memory.memory_id} was deleted.`);
      } catch (caught) {
        setNotice(
          caught instanceof Error
            ? caught.message
            : "The memory could not be deleted.",
        );
        if (caught instanceof ApiError && caught.status === 404) {
          await refresh(query);
        }
      } finally {
        setDeletingId(null);
      }
    },
    [query, refresh],
  );

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Memory"
        description="Facts the assistant explicitly remembers per user — preferences, customer and business context, standing instructions — stored in PostgreSQL so they survive restarts."
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
        <IconSparkles className="mt-0.5 h-4.5 w-4.5 shrink-0 text-indigo-500" />
        <p className="leading-relaxed">
          Memories are saved only when a user explicitly asks the assistant to
          remember something, and they are context only — memory never executes
          business actions. Owner keys are application-level scopes, not
          authentication.
        </p>
      </div>

      {/* Owner scope picker */}
      <form
        className="mt-6 flex flex-wrap items-end gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          setQuery(ownerKey.trim() || "default");
        }}
      >
        <label className="flex flex-col gap-1 text-xs font-medium text-slate-600">
          Owner key
          <input
            value={ownerKey}
            onChange={(event) => setOwnerKey(event.target.value)}
            placeholder="default"
            spellCheck={false}
            className="w-56 rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono text-sm text-slate-900 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </label>
        <button
          type="submit"
          className="rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
        >
          Load
        </button>
      </form>

      {/* Feedback notice */}
      {notice && (
        <p
          role="status"
          aria-live="polite"
          className="mt-4 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700"
        >
          {notice}
        </p>
      )}

      {/* Initial loading */}
      {initialLoading && (
        <div className="mt-6 space-y-3" aria-label="Loading memories">
          {[0, 1, 2].map((n) => (
            <div
              key={n}
              className="h-20 animate-pulse rounded-xl border border-slate-200 bg-white"
            />
          ))}
        </div>
      )}

      {/* Load error */}
      {!initialLoading && loadError && (
        <div className="mt-6" role="alert">
          <EmptyState
            icon={<IconAlert className="h-6 w-6" />}
            title="Could not load memories"
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
      {!initialLoading && !loadError && memories.length === 0 && (
        <div className="mt-6">
          <EmptyState
            icon={<IconSparkles className="h-6 w-6" />}
            title="No memories for this owner"
            description="Nothing has been saved for this owner key yet. Ask the assistant in Chat to remember a preference, customer detail, or standing instruction."
          />
        </div>
      )}

      {/* List */}
      {!initialLoading && !loadError && memories.length > 0 && (
        <div className="mt-6 space-y-3">
          <p className="text-xs font-medium text-slate-500">
            {memories.length} memor{memories.length === 1 ? "y" : "ies"} for{" "}
            <span className="font-mono text-slate-700">{query}</span>
          </p>
          {memories.map((memory) => (
            <article
              key={memory.memory_id}
              className="flex items-start gap-4 rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm"
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[11px] font-semibold ${
                      TYPE_STYLES[memory.memory_type] ??
                      "border-slate-200 bg-slate-50 text-slate-700"
                    }`}
                  >
                    {memory.memory_type}
                  </span>
                  <span className="font-mono text-[11px] text-slate-400">
                    #{memory.memory_id} · {memory.owner_key}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-slate-800">
                  {memory.content}
                </p>
                <p className="mt-1.5 text-xs text-slate-500">
                  Created {formatBackendTimestamp(memory.created_at)} · Updated{" "}
                  {formatBackendTimestamp(memory.updated_at)}
                </p>
              </div>
              <button
                type="button"
                disabled={deletingId !== null}
                onClick={() => void handleDelete(memory)}
                aria-label={`Delete memory ${memory.memory_id}`}
                className="shrink-0 rounded-lg border border-slate-300 bg-white p-2 text-slate-500 transition-colors hover:border-rose-300 hover:bg-rose-50 hover:text-rose-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <IconX className="h-4 w-4" />
              </button>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
