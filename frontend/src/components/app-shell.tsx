"use client";

/**
 * Application chrome: brand sidebar (drawer on mobile), top bar with
 * the live backend connection badge, and the page content area.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { BackendStatusBadge } from "@/components/backend-status-badge";
import {
  IconBoxes,
  IconCart,
  IconChart,
  IconChat,
  IconMenu,
  IconPulse,
  IconShield,
  IconSparkles,
  IconX,
} from "@/components/icons";
import { usePendingApprovalCount } from "@/hooks/use-pending-approval-count";
import { API_HOST_LABEL } from "@/lib/api";

type NavItem = {
  href:
    | "/chat"
    | "/inventory"
    | "/orders"
    | "/reports"
    | "/approvals"
    | "/memory"
    | "/observability";
  label: string;
  icon: (props: { className?: string }) => React.ReactElement;
  badge?: "approvals";
};

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Assistant",
    items: [
      { href: "/chat", label: "Chat", icon: IconChat },
      { href: "/memory", label: "Memory", icon: IconSparkles },
    ],
  },
  {
    label: "Operations",
    items: [
      { href: "/inventory", label: "Inventory", icon: IconBoxes },
      { href: "/orders", label: "Orders", icon: IconCart },
    ],
  },
  {
    label: "Insights",
    items: [{ href: "/reports", label: "Reports", icon: IconChart }],
  },
  {
    label: "Governance",
    items: [
      { href: "/approvals", label: "Approvals", icon: IconShield, badge: "approvals" },
      { href: "/observability", label: "Observability", icon: IconPulse },
    ],
  },
];

const ALL_NAV_ITEMS = NAV_GROUPS.flatMap((group) => group.items);

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const pendingCount = usePendingApprovalCount();

  // Close the mobile drawer whenever the route changes (state adjusted
  // during render — the React-endorsed alternative to an effect).
  const [lastPathname, setLastPathname] = useState(pathname);
  if (lastPathname !== pathname) {
    setLastPathname(pathname);
    setDrawerOpen(false);
  }

  // Escape closes the drawer, as overlay dialogs should.
  useEffect(() => {
    if (!drawerOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setDrawerOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawerOpen]);

  const current =
    ALL_NAV_ITEMS.find((item) => item.href === pathname) ?? undefined;

  return (
    <div className="flex h-dvh overflow-hidden bg-slate-100 text-slate-900">
      {/* Mobile overlay behind the drawer */}
      {drawerOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-950/60 backdrop-blur-sm lg:hidden"
          onClick={() => setDrawerOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        aria-label="Main navigation"
        className={`fixed inset-y-0 left-0 z-50 flex w-72 shrink-0 flex-col bg-slate-950 transition-transform duration-200 ease-in-out lg:static lg:translate-x-0 ${
          drawerOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Brand */}
        <div className="flex items-center justify-between px-5 pb-4 pt-6">
          <Link href="/chat" className="group flex items-center gap-3">
            <span
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-400 to-violet-600 text-base font-bold text-white shadow-lg shadow-indigo-950/50"
              aria-hidden="true"
            >
              D
            </span>
            <span>
              <span className="block text-[15px] font-semibold tracking-tight text-white">
                DibantuAI
              </span>
              <span className="block text-xs text-slate-400">
                AI Business Assistant
              </span>
            </span>
          </Link>
          <button
            type="button"
            onClick={() => setDrawerOpen(false)}
            aria-label="Close navigation"
            className="rounded-lg p-1.5 text-slate-400 hover:bg-white/5 hover:text-white lg:hidden"
          >
            <IconX className="h-5 w-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-6 overflow-y-auto px-3 py-2">
          {NAV_GROUPS.map((group) => (
            <div key={group.label}>
              <p className="px-3 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                {group.label}
              </p>
              <ul className="space-y-1">
                {group.items.map((item) => {
                  const active = pathname === item.href;
                  const Icon = item.icon;
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        aria-current={active ? "page" : undefined}
                        className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                          active
                            ? "bg-indigo-500/15 text-white"
                            : "text-slate-400 hover:bg-white/5 hover:text-white"
                        }`}
                      >
                        <Icon
                          className={`h-[18px] w-[18px] ${
                            active ? "text-indigo-300" : "text-slate-500"
                          }`}
                        />
                        {item.label}
                        {item.badge === "approvals" &&
                          pendingCount !== null &&
                          pendingCount > 0 && (
                            <span className="ml-auto rounded-full bg-indigo-500/20 px-2 py-0.5 text-xs font-semibold text-indigo-200">
                              {pendingCount}
                            </span>
                          )}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>

        {/* Sidebar footer */}
        <div className="border-t border-white/5 px-5 py-4">
          <p className="text-[11px] leading-relaxed text-slate-500">
            FastAPI backend
            <span className="block font-mono text-slate-400">
              {API_HOST_LABEL}
            </span>
          </p>
        </div>
      </aside>

      {/* Content column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center gap-3 border-b border-slate-200 bg-white/80 px-4 backdrop-blur sm:px-6">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open navigation"
            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden"
          >
            <IconMenu className="h-5 w-5" />
          </button>

          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900">
              {current?.label ?? "DibantuAI"}
            </p>
            <p className="hidden truncate text-xs text-slate-500 sm:block">
              DibantuAI · AI Agent for Business Operations
            </p>
          </div>

          <div className="ml-auto">
            <BackendStatusBadge />
          </div>
        </header>

        <main className="flex min-h-0 flex-1 flex-col">{children}</main>
      </div>
    </div>
  );
}
