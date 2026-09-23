"use client";

/**
 * Reports page. There is no report endpoint on the backend yet —
 * sales summaries are generated on demand by the AI assistant from
 * its business tools via POST /api/chat. The cards below fire real
 * assistant queries; nothing on this page is simulated data.
 */

import { useRef } from "react";

import {
  AssistantAsk,
  type AssistantAskHandle,
} from "@/components/assistant-ask";
import { PageHeader } from "@/components/page-header";
import { IconChart, IconSparkles } from "@/components/icons";

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

export function ReportsView() {
  const askRef = useRef<AssistantAskHandle>(null);

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Reports"
        description="Sales reports are generated on demand by the AI assistant. A dedicated reporting API will add stored, exportable reports here later."
      />

      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {REPORTS.map((report) => (
          <button
            key={report.id}
            type="button"
            onClick={() => askRef.current?.ask(report.prompt)}
            className="group rounded-xl border border-slate-200 bg-white p-5 text-left shadow-sm transition-colors hover:border-indigo-300 hover:bg-indigo-50/40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-500 transition-colors group-hover:bg-indigo-100 group-hover:text-indigo-600">
              <IconChart className="h-4.5 w-4.5" />
            </span>
            <h3 className="mt-3 text-sm font-semibold text-slate-900">
              {report.label} sales
            </h3>
            <p className="mt-1 text-sm leading-relaxed text-slate-500">
              {report.description}
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

      <div className="mt-6 rounded-xl border border-slate-200 bg-white px-5 py-4 text-sm leading-relaxed text-slate-600 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-900">
          Planned: stored reports
        </h2>
        <p className="mt-1.5 text-slate-500">
          Saved, exportable reports will appear here once the backend exposes a
          reporting API. Until then, every number you see above is fetched live
          from the assistant — nothing on this page is cached or simulated.
        </p>
      </div>
    </div>
  );
}
