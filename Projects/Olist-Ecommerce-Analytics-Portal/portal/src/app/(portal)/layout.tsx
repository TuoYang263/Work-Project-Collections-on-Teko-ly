import type { ReactNode } from "react"

import { AppSidebar } from "@/components/portal/app-sidebar"
import { Badge } from "@/components/ui/badge"

export default function PortalLayout({
  children,
}: Readonly<{
  children: ReactNode
}>) {
  return (
    <div className="flex min-h-svh bg-muted/20">
      <AppSidebar />

      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-30 flex h-[72px] items-center justify-between border-b bg-background/90 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/75 md:px-6 lg:px-8">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold tracking-tight">
              E-Commerce Analytics & Pipeline Monitoring
            </div>
            <div className="truncate text-xs text-muted-foreground">
              Olist data platform · Operational workspace
            </div>
          </div>

          <div className="hidden items-center gap-2 sm:flex">
            <Badge
              variant="outline"
              className="gap-1.5 rounded-full px-2.5 py-1 font-normal"
            >
              <span className="size-1.5 rounded-full bg-emerald-500" />
              Operational
            </Badge>

            <Badge
              variant="secondary"
              className="rounded-full px-2.5 py-1 font-normal"
            >
              BigQuery · EU
            </Badge>
          </div>
        </header>

        <main className="px-4 py-6 md:px-6 lg:px-8">
          <div className="mx-auto w-full max-w-[1600px]">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
