"use client";

import { usePathname } from "next/navigation";
import { navSections } from "@/lib/nav";

function titleForPath(pathname: string) {
  for (const section of navSections) {
    for (const item of section.items) {
      if (item.href === pathname) return item.label;
      if (item.href !== "/" && pathname.startsWith(item.href)) return item.label;
    }
  }
  return "Dashboard";
}

export default function Topbar() {
  const pathname = usePathname();
  const title = titleForPath(pathname);

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-6">
      <div>
        <h1 className="text-sm font-semibold text-foreground">{title}</h1>
        <p className="text-xs text-muted">Authenticated workspace</p>
      </div>
      <div className="flex items-center gap-3">
        <span className="rounded-full bg-background px-3 py-1 text-xs text-muted">
          user_id: demo-user
        </span>
      </div>
    </header>
  );
}
