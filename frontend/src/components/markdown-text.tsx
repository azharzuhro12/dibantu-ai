"use client";

/**
 * Renders assistant replies as Markdown (bullet/numbered lists, line
 * breaks, inline code — plus styled, scrollable GFM tables).
 *
 * House style: assistant content is ALWAYS normal font weight. Markdown
 * `**text**` must not show literal asterisks, but it must not render
 * bold either — so `strong` is overridden to a normal-weight span, and
 * headings/table headers (which browsers bold by default) are mapped
 * to normal weight as well.
 *
 * Safety: react-markdown parses to React elements and ignores raw HTML
 * by default (no rehype-raw), so model output can never inject markup
 * or scripts. User messages are intentionally NOT routed through this
 * component — they always render as plain text.
 */

import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

/** All headings render as normal-weight paragraphs (no bold, no resize). */
function heading({ children }: { children?: React.ReactNode }) {
  return <p className="mb-2 font-normal last:mb-0">{children}</p>;
}

export function MarkdownText({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkBreaks]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => (
          <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0">
            {children}
          </ol>
        ),
        li: ({ children }) => <li className="pl-1">{children}</li>,
        // `**text**` renders as normal text: no literal **, no bold.
        strong: ({ children }) => (
          <span className="font-normal">{children}</span>
        ),
        em: ({ children }) => <em className="italic">{children}</em>,
        h1: heading,
        h2: heading,
        h3: heading,
        h4: heading,
        h5: heading,
        h6: heading,
        // GFM tables: semantic table markup inside a bordered, rounded
        // container that scrolls horizontally when the chat is narrow.
        // All cells stay normal weight (`---:` alignment is preserved
        // via the forwarded style prop).
        table: ({ children }) => (
          <div className="my-3 max-w-full overflow-x-auto rounded-lg border border-slate-200 first:mt-0 last:mb-0">
            <table className="w-full border-collapse text-[13px]">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-slate-50">{children}</thead>
        ),
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => (
          <tr className="border-t border-slate-200 first:border-t-0">
            {children}
          </tr>
        ),
        th: ({ style, children }) => (
          <th
            style={style}
            className="whitespace-nowrap px-3 py-2 text-left font-normal text-slate-600"
          >
            {children}
          </th>
        ),
        td: ({ style, children }) => (
          <td style={style} className="whitespace-nowrap px-3 py-2 align-top">
            {children}
          </td>
        ),
        code: ({ className, children }) => (
          <code
            className={
              className?.includes("language-")
                ? "font-mono text-[13px]"
                : "rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[13px] text-indigo-700"
            }
          >
            {children}
          </code>
        ),
        pre: ({ children }) => (
          <pre className="mb-2 overflow-x-auto rounded-lg bg-slate-900 p-3 text-[13px] text-slate-100 last:mb-0">
            {children}
          </pre>
        ),
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="text-indigo-600 underline hover:text-indigo-500"
          >
            {children}
          </a>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
