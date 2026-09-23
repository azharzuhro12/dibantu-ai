"use client";

/**
 * Reusable "ask the assistant" panel. Inventory, Orders, and Reports
 * have no dedicated backend endpoints yet — their data lives behind
 * the AI agent — so these pages query the real assistant through
 * POST /api/chat instead of showing fabricated tables.
 *
 * The panel is explicitly labeled as AI answers, never as live
 * database records. Parents that need to trigger a question
 * programmatically (e.g. the report cards) use the imperative handle:
 * `askRef.current?.ask(prompt)`.
 */

import {
  useCallback,
  useImperativeHandle,
  useRef,
  useState,
  type Ref,
} from "react";

import { IconAlert, IconSend, IconSparkles } from "@/components/icons";
import { MarkdownText } from "@/components/markdown-text";
import { sendMessage } from "@/lib/api";
import { formatTime } from "@/lib/format";

type AssistantAnswer = {
  question: string;
  answer: string;
  answeredAt: number;
};

/** Imperative API exposed to parents via a ref. */
export type AssistantAskHandle = {
  ask: (prompt: string) => void;
};

export function AssistantAsk({
  placeholder = "Ask about your business…",
  suggestions = [],
  ref,
}: {
  placeholder?: string;
  suggestions?: string[];
  ref?: Ref<AssistantAskHandle>;
}) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AssistantAnswer | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [asked, setAsked] = useState(false);
  const loadingRef = useRef(false);

  const ask = useCallback(async (prompt: string) => {
    const trimmed = prompt.trim();
    if (!trimmed || loadingRef.current) {
      return;
    }
    loadingRef.current = true;
    setLoading(true);
    setError(null);
    setResult(null);
    setAsked(true);
    try {
      const response = await sendMessage(trimmed);
      setResult({
        question: trimmed,
        answer: response.response,
        answeredAt: Date.now(),
      });
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The assistant could not be reached.",
      );
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, []);

  useImperativeHandle(
    ref,
    () => ({ ask: (prompt: string) => void ask(prompt) }),
    [ask],
  );

  return (
    <section
      aria-label="Ask the DibantuAI assistant"
      className="rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <header className="flex items-start gap-3 border-b border-slate-100 px-5 py-4">
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
          <IconSparkles className="h-4 w-4" />
        </span>
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Ask DibantuAI
          </h2>
          <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
            Answered live by the AI assistant (POST /api/chat) — not a stored
            report.
          </p>
        </div>
      </header>

      <div className="space-y-4 px-5 py-4">
        {/* Answer / loading / error area */}
        <div aria-live="polite" className="min-h-[3rem]">
          {loading && (
            <div className="flex items-center gap-3 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-500">
              <TypingDots />
              Assistant is checking your business data…
            </div>
          )}

          {!loading && error && (
            <div
              role="alert"
              className="flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
            >
              <IconAlert className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!loading && !error && result && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
              <p className="text-xs font-medium text-slate-500">
                You asked · {formatTime(result.answeredAt)}
              </p>
              <p className="mt-1 text-sm font-medium text-slate-800">
                {result.question}
              </p>
              <div className="mt-2 text-slate-700">
                <MarkdownText>{result.answer}</MarkdownText>
              </div>
            </div>
          )}

          {!loading && !error && !result && !asked && (
            <p className="px-1 py-2 text-sm text-slate-400">
              Ask anything about stock, orders, customers, or sales — the
              assistant answers from live business tools.
            </p>
          )}
        </div>

        {/* Suggestion chips */}
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                disabled={loading}
                onClick={() => void ask(suggestion)}
                className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}

        {/* Input row */}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void ask(input);
            setInput("");
          }}
          className="flex items-center gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={placeholder}
            aria-label="Message for the DibantuAI assistant"
            disabled={loading}
            className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-3.5 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-400"
          />
          <button
            type="submit"
            disabled={loading || input.trim().length === 0}
            aria-label="Send message to assistant"
            className="flex h-[42px] w-[42px] shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-white transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            <IconSend className="h-4 w-4" />
          </button>
        </form>
      </div>
    </section>
  );
}

/** Three bouncing dots shown while the assistant is working. */
export function TypingDots() {
  return (
    <span className="flex items-center gap-1" aria-hidden="true">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="h-1.5 w-1.5 animate-[typing-bounce_1.2s_ease-in-out_infinite] rounded-full bg-indigo-400"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </span>
  );
}
