"use client";

/**
 * Lightweight poll of GET /api/approvals used only for the sidebar
 * badge. The Approvals page fetches the full list itself; this hook
 * stays silent on errors (a down backend simply hides the badge).
 */

import { useEffect, useState } from "react";

import { listApprovals } from "@/lib/api";

const POLL_INTERVAL_MS = 30_000;

export function usePendingApprovalCount(): number | null {
  const [count, setCount] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;

    const poll = async () => {
      try {
        const response = await listApprovals();
        if (alive) {
          setCount(response.count);
        }
      } catch {
        if (alive) {
          setCount(null);
        }
      }
    };

    const first = setTimeout(() => void poll(), 0);
    const interval = setInterval(() => {
      if (!document.hidden) {
        void poll();
      }
    }, POLL_INTERVAL_MS);

    return () => {
      alive = false;
      clearTimeout(first);
      clearInterval(interval);
    };
  }, []);

  return count;
}
