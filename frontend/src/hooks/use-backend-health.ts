"use client";

/**
 * Polls GET /health so the UI can show a live backend connection badge.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { checkHealth } from "@/lib/api";

export type BackendHealthState = {
  /** "checking" until the first probe resolves, then online/offline. */
  state: "checking" | "online" | "offline";
  /** Backend version from /health, shown when online. */
  version: string | null;
  /** Human-readable reason while offline. */
  error: string | null;
};

const POLL_INTERVAL_MS = 15_000;

export function useBackendHealth(): BackendHealthState {
  const [health, setHealth] = useState<BackendHealthState>({
    state: "checking",
    version: null,
    error: null,
  });
  const alive = useRef(true);

  const probe = useCallback(async () => {
    try {
      const response = await checkHealth();
      if (alive.current) {
        setHealth({ state: "online", version: response.version, error: null });
      }
    } catch (error) {
      if (alive.current) {
        setHealth({
          state: "offline",
          version: null,
          error: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }
  }, []);

  useEffect(() => {
    alive.current = true;

    // First probe runs as a scheduled callback (not synchronously in
    // the effect body), then on an interval while the tab is visible.
    const first = setTimeout(() => void probe(), 0);
    const interval = setInterval(() => {
      if (!document.hidden) {
        void probe();
      }
    }, POLL_INTERVAL_MS);

    const onVisible = () => {
      if (!document.hidden) {
        void probe();
      }
    };
    document.addEventListener("visibilitychange", onVisible);

    return () => {
      alive.current = false;
      clearTimeout(first);
      clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [probe]);

  return health;
}
