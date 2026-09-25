/** Small formatting helpers shared across pages. */

/** Format a time as a short locale time, e.g. "14:05" / "2:05 PM". */
export function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp);
}

/**
 * Format a backend ISO timestamp (e.g. "2026-09-23T14:05:00") as a
 * readable date-time. Falls back to the raw string when it cannot be
 * parsed — the UI never shows "Invalid Date".
 */
export function formatBackendTimestamp(iso: string | null): string {
  if (!iso) {
    return "—";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

/** Humanize an approval action, e.g. "refund_order" -> "Refund order". */
export function humanizeAction(action: string): string {
  return action
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/** Format an IDR amount, e.g. 18000 -> "Rp18.000". */
export function formatRupiah(amount: number): string {
  return `Rp${new Intl.NumberFormat("id-ID").format(amount)}`;
}
