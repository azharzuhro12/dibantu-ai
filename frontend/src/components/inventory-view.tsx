"use client";

/**
 * Inventory page. There is no dedicated inventory endpoint on the
 * backend yet — stock data lives behind the AI agent — so this page
 * (a) queries the real assistant through POST /api/chat and (b) shows
 * the table shape a future GET /api/inventory will fill in.
 */

import { AssistantAsk } from "@/components/assistant-ask";
import { ComingEndpointPanel } from "@/components/coming-endpoint-panel";
import { PageHeader } from "@/components/page-header";
import { IconBoxes } from "@/components/icons";

const CAPABILITIES = [
  "Check any product's current stock",
  "See which products are running low",
  "Adjust stock levels (bulk updates need approval)",
];

export function InventoryView() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Inventory"
        description="Stock levels are served by the AI assistant in real time. A dedicated inventory API will populate the table below once the backend exposes it."
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
        <ComingEndpointPanel
          title="Live stock table"
          endpoint="GET /api/inventory"
          columns={["Product", "In stock", "Status"]}
          note="This panel is intentionally empty: no inventory endpoint exists on the backend yet, so no rows are shown. Once it does, this table fills in automatically."
        />
      </div>
    </div>
  );
}
