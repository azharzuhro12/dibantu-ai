"use client";

/**
 * ChatGPT-style chat interface for the DibantuAI agent (POST /api/chat).
 *
 * - Enter sends, Shift+Enter inserts a newline
 * - Auto-scrolls to the newest message (unless the reader scrolled up)
 * - Loading, error, and per-message retry states
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { IconAlert, IconRefresh, IconSend, IconSparkles } from "@/components/icons";
import { MarkdownText } from "@/components/markdown-text";
import { sendMessage } from "@/lib/api";
import { formatTime } from "@/lib/format";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  /** Failed assistant turn; `retryWith` holds the user text to resend. */
  error?: boolean;
  retryWith?: string;
};

const SUGGESTIONS = [
  "Cek stok kopi susu",
  "Laporan penjualan harian",
  "Produk apa saja yang stoknya rendah?",
  "Cari customer Budi",
];

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Keep pinned to the bottom while the reader is near the bottom.
  const handleListScroll = useCallback(() => {
    const el = listRef.current;
    if (!el) {
      return;
    }
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distance < 120;
  }, []);

  useEffect(() => {
    if (stickToBottomRef.current && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, loading]);

  // Auto-grow the textarea up to ~6 lines.
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) {
      return;
    }
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, []);

  const runAssistant = useCallback(
    async (userText: string, replaceErrorId?: string) => {
      setLoading(true);
      stickToBottomRef.current = true;
      try {
        const response = await sendMessage(userText);
        const reply: ChatMessage = {
          id: newId(),
          role: "assistant",
          content: response.response,
          createdAt: Date.now(),
        };
        setMessages((current) =>
          replaceErrorId
            ? current.map((m) => (m.id === replaceErrorId ? reply : m))
            : [...current, reply],
        );
      } catch (caught) {
        const message =
          caught instanceof Error
            ? caught.message
            : "Something went wrong talking to the assistant.";
        const failure: ChatMessage = {
          id: newId(),
          role: "assistant",
          content: message,
          createdAt: Date.now(),
          error: true,
          retryWith: userText,
        };
        setMessages((current) =>
          replaceErrorId
            ? current.map((m) => (m.id === replaceErrorId ? failure : m))
            : [...current, failure],
        );
      } finally {
        setLoading(false);
        textareaRef.current?.focus();
      }
    },
    [],
  );

  const send = useCallback(
    (rawText: string) => {
      const text = rawText.trim();
      if (!text || loading) {
        return;
      }
      setMessages((current) => [
        ...current,
        { id: newId(), role: "user", content: text, createdAt: Date.now() },
      ]);
      setInput("");
      requestAnimationFrame(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto";
        }
      });
      void runAssistant(text);
    },
    [loading, runAssistant],
  );

  const retry = useCallback(
    (message: ChatMessage) => {
      if (!message.retryWith || loading) {
        return;
      }
      void runAssistant(message.retryWith, message.id);
    },
    [loading, runAssistant],
  );

  const empty = messages.length === 0;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Messages */}
      <div
        ref={listRef}
        onScroll={handleListScroll}
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
        className="min-h-0 flex-1 overflow-y-auto"
      >
        <div className="mx-auto w-full max-w-3xl px-4 py-6 sm:px-6">
          {empty ? (
            <div className="flex flex-col items-center justify-center py-16 text-center sm:py-24">
              <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-violet-600 text-white shadow-lg shadow-indigo-500/25">
                <IconSparkles className="h-7 w-7" />
              </span>
              <h2 className="mt-5 text-xl font-semibold tracking-tight text-slate-900">
                How can I help your business today?
              </h2>
              <p className="mt-2 max-w-md text-sm leading-relaxed text-slate-500">
                Ask about stock, orders, customers, or sales — DibantuAI answers
                from your live business tools and asks for approval before any
                sensitive action.
              </p>
              <div className="mt-8 grid w-full max-w-lg gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => send(suggestion)}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 shadow-sm transition-colors hover:border-indigo-300 hover:bg-indigo-50/50 hover:text-indigo-800"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <ul className="space-y-6">
              {messages.map((message) => (
                <li
                  key={message.id}
                  className={`flex gap-3 ${
                    message.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {message.role === "assistant" && (
                    <span
                      aria-hidden="true"
                      className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-xs font-bold text-white"
                    >
                      D
                    </span>
                  )}
                  <div
                    className={`flex max-w-[85%] flex-col sm:max-w-[75%] ${
                      message.role === "user" ? "items-end" : "items-start"
                    }`}
                  >
                    <div
                      className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                        message.role === "user"
                          ? "rounded-br-md bg-indigo-600 text-white"
                          : message.error
                            ? "rounded-bl-md border border-rose-200 bg-rose-50 text-rose-700"
                            : "rounded-bl-md border border-slate-200 bg-white text-slate-800 shadow-sm"
                      }`}
                    >
                      {message.role === "assistant" && message.error && (
                        <span className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide">
                          <IconAlert className="h-3.5 w-3.5" />
                          Request failed
                        </span>
                      )}
                      {message.role === "assistant" && !message.error ? (
                        <MarkdownText>{message.content}</MarkdownText>
                      ) : (
                        <span className="whitespace-pre-wrap">
                          {message.content}
                        </span>
                      )}
                      {message.role === "assistant" && message.error && (
                        <button
                          type="button"
                          onClick={() => retry(message)}
                          disabled={loading}
                          className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-rose-300 bg-white px-3 py-1.5 text-xs font-semibold text-rose-700 transition-colors hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          <IconRefresh className="h-3.5 w-3.5" />
                          Retry
                        </button>
                      )}
                    </div>
                    <time
                      dateTime={new Date(message.createdAt).toISOString()}
                      className="mt-1 px-1 text-[11px] text-slate-400"
                    >
                      {formatTime(message.createdAt)}
                    </time>
                  </div>
                </li>
              ))}

              {loading && (
                <li className="flex gap-3" aria-label="Assistant is typing">
                  <span
                    aria-hidden="true"
                    className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 text-xs font-bold text-white"
                  >
                    D
                  </span>
                  <div className="flex items-center gap-3 rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3.5 shadow-sm">
                    <span className="flex items-center gap-1" aria-hidden="true">
                      {[0, 150, 300].map((delay) => (
                        <span
                          key={delay}
                          className="h-1.5 w-1.5 animate-[typing-bounce_1.2s_ease-in-out_infinite] rounded-full bg-indigo-400"
                          style={{ animationDelay: `${delay}ms` }}
                        />
                      ))}
                    </span>
                    <span className="text-xs text-slate-400">
                      DibantuAI is thinking…
                    </span>
                  </div>
                </li>
              )}
            </ul>
          )}
        </div>
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t border-slate-200 bg-white/90 backdrop-blur">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            send(input);
          }}
          className="mx-auto w-full max-w-3xl px-4 py-4 sm:px-6"
        >
          <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition-colors focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500/20">
            <label htmlFor="chat-input" className="sr-only">
              Message DibantuAI
            </label>
            <textarea
              id="chat-input"
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(event) => {
                setInput(event.target.value);
                resizeTextarea();
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send(input);
                }
              }}
              placeholder="Ask about stock, orders, customers, sales…"
              className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent px-2.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading || input.trim().length === 0}
              aria-label="Send message"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition-colors hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              <IconSend className="h-4.5 w-4.5" />
            </button>
          </div>
          <p className="mt-2 px-1 text-center text-[11px] text-slate-400">
            Enter to send · Shift+Enter for a new line
          </p>
        </form>
      </div>
    </div>
  );
}
