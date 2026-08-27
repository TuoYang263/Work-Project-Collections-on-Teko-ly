"use client"

import {
  ChartNoAxesCombined,
  Database,
  LayoutDashboard,
  ShieldCheck,
} from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const navigation = [
  {
    href: "/overview",
    label: "Overview",
    description: "Operational state",
    icon: LayoutDashboard,
  },
  {
    href: "/analytics",
    label: "Analytics",
    description: "Business decisions",
    icon: ChartNoAxesCombined,
  },
  {
    href: "/reliability",
    label: "Reliability",
    description: "Quality and findings",
    icon: ShieldCheck,
  },
]

export function AppSidebar() {
  const pathname = usePathname()

  return (
    <aside className="sticky top-0 hidden h-svh w-[280px] shrink-0 border-r bg-sidebar lg:flex lg:flex-col">
      <div className="flex h-[72px] items-center gap-3 border-b px-4">
        <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
          <Database className="size-4" />
        </div>

        <div className="min-w-0">
          <div className="truncate text-sm font-semibold tracking-tight">
            Olist Portal
          </div>
          <div className="truncate text-xs text-muted-foreground">
            Analytics & Reliability
          </div>
        </div>
      </div>

      <div className="flex-1 px-3 py-5">
        <div className="px-3 pb-2 text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground">
          Workspace
        </div>

        <nav className="space-y-1.5">
          {navigation.map((item) => {
            const isActive =
              pathname === item.href ||
              pathname.startsWith(`${item.href}/`)
            const Icon = item.icon

            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={isActive ? "page" : undefined}
                className={`group flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors ${
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-sm ring-1 ring-sidebar-border"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/60"
                }`}
              >
                <div
                  className={`flex size-9 shrink-0 items-center justify-center rounded-lg transition-colors ${
                    isActive
                      ? "bg-background text-foreground shadow-sm"
                      : "bg-muted/60 text-muted-foreground group-hover:text-foreground"
                  }`}
                >
                  <Icon className="size-4" />
                </div>

                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">
                    {item.label}
                  </div>
                  <div className="truncate text-xs text-muted-foreground">
                    {item.description}
                  </div>
                </div>
              </Link>
            )
          })}
        </nav>
      </div>

      <div className="border-t p-4">
        <div className="rounded-xl border bg-background/70 p-3 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="text-xs font-medium">
                Data platform
              </div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">
                BigQuery · EU
              </div>
            </div>

            <div className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-700">
              <span className="size-1.5 rounded-full bg-emerald-500" />
              Connected
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}
