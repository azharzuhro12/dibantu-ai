/**
 * Placeholder panel for pages whose dedicated backend endpoint does not
 * exist yet. It shows the table *shape* the page is designed for, with
 * deliberately empty rows — no fabricated business data.
 */

import { IconSparkles } from "@/components/icons";

export function ComingEndpointPanel({
  title,
  endpoint,
  columns,
  note,
}: {
  title: string;
  endpoint: string;
  columns: string[];
  note: string;
}) {
  return (
    <section
      aria-label={`${title} awaiting backend endpoint`}
      className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <header className="flex flex-col gap-1 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
        <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-dashed border-slate-300 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-500">
          <span className="h-1.5 w-1.5 rounded-full bg-slate-300" aria-hidden="true" />
          Awaiting endpoint · <code className="font-mono">{endpoint}</code>
        </span>
      </header>

      <div role="presentation" aria-hidden="true">
        <div className="grid gap-4 border-b border-slate-100 bg-slate-50/60 px-5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400"
          style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))` }}
        >
          {columns.map((column) => (
            <span key={column}>{column}</span>
          ))}
        </div>
        {[0, 1, 2].map((row) => (
          <div
            key={row}
            className="grid gap-4 border-b border-slate-50 px-5 py-3.5 last:border-b-0"
            style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))` }}
          >
            {columns.map((column) => (
              <span key={column} className=" h-3 rounded bg-slate-100" />
            ))}
          </div>
        ))}
      </div>

      <p className="flex items-start gap-2.5 border-t border-slate-100 bg-slate-50/60 px-5 py-3.5 text-xs leading-relaxed text-slate-500">
        <IconSparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-400" />
        {note}
      </p>
    </section>
  );
}
