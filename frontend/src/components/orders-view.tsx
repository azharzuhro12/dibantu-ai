"use client";

/**
 * Orders page. Order management currently happens through the AI
 * assistant and the existing backend; there is no dedicated orders
 * endpoint yet, so this page offers assistant-driven ordering plus an
 * honest placeholder for the future orders API.
 */

import Link from "next/link";

import { AssistantAsk } from "@/components/assistant-ask";
import { ComingEndpointPanel } from "@/components/coming-endpoint-panel";
import { PageHeader } from "@/components/page-header";
import { IconArrowRight, IconCart } from "@/components/icons";

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

export function OrdersView() {
  return (
    <div className="mx-auto w-full max-w-5xl flex-1 overflow-y-auto px-4 py-8 sm:px-6 lg:px-8">
      <PageHeader
        title="Orders"
        description="Order management is handled through the AI assistant today. Use the prompts below, or open the full chat for anything more complex."
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
        <ComingEndpointPanel
          title="Order history"
          endpoint="GET /api/orders"
          columns={["Order", "Customer", "Items", "Total"]}
          note="No orders endpoint exists on the backend yet, so this table stays empty rather than showing sample data. Order activity is visible today by asking the assistant."
        />
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
