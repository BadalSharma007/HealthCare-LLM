"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { navSections } from "@/lib/nav";

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-64 shrink-0 flex-col bg-sidebar text-sidebar-text">
      <div className="border-b border-white/10 px-5 py-5">
        <Link href="/" className="text-base font-semibold text-white">
          Resume<span className="text-sky-400">Gatherer</span>
        </Link>
        <p className="mt-1 text-xs text-slate-400">Dashboard</p>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {navSections.map((section) => (
          <div key={section.title} className="mb-5">
            <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              {section.title}
            </p>
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const active = isActive(pathname, item.href);
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className={`block rounded-lg px-3 py-2 text-sm transition ${
                        active
                          ? "bg-white/10 font-medium text-sidebar-active"
                          : "hover:bg-white/5 hover:text-white"
                      }`}
                    >
                      {item.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="border-t border-white/10 px-4 py-4 text-xs text-slate-500">
        API: {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
      </div>
    </aside>
  );
}
