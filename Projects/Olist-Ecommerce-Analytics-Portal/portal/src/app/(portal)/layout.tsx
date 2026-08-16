import type { ReactNode } from "react";

import { AppSidebar } from "@/components/portal/app-sidebar";

export default function PortalLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <div className="flex min-h-screen bg-muted/30">
      <AppSidebar />

      <div className="min-w-0 flex-1">
        <header className="flex h-16 items-center border-b bg-background px-6">
          <div>
            <div className="text-sm font-medium">
              E-Commerce Analytics & Pipeline Monitoring
            </div>
            <div className="text-xs text-muted-foreground">
              Operational Portal
            </div>
          </div>
        </header>

        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}