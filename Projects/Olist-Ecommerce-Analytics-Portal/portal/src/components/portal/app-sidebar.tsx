"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  {
    href: "/overview",
    label: "Overview",
    description: "Operational summary",
  },
  {
    href: "/analytics",
    label: "Analytics",
    description: "Business and geospatial analytics",
  },
  {
    href: "/reliability",
    label: "Reliability",
    description: "Pipeline quality and findings",
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 shrink-0 border-r bg-background">
      <div className="flex h-16 items-center border-b px-5">
        <div>
          <div className="text-sm font-semibold">Olist Portal</div>
          <div className="text-xs text-muted-foreground">
            Analytics & Reliability
          </div>
        </div>
      </div>

      <nav className="space-y-1 p-3">
        {navigation.map((item) => {
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? "page" : undefined}
              className={`block rounded-md px-3 py-2 transition-colors ${
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "hover:bg-accent/60"
              }`}
            >
              <div className="text-sm font-medium">{item.label}</div>
              <div className="text-xs text-muted-foreground">
                {item.description}
              </div>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
