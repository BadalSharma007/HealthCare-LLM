import Link from "next/link";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { navSections } from "@/lib/nav";

const quickLinks = navSections
  .flatMap((s) => s.items)
  .filter((i) => i.href !== "/")
  .slice(0, 8);

export default function DashboardHomePage() {
  return (
    <div>
      <PageHeader
        title="Dashboard overview"
        description="Authenticated app for resume intelligence, profile enrichment, job matching, and generation — backed by Resume-Data_Gather."
      />

      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Resume modules", value: "5 pages" },
          { label: "Profile sources", value: "GitHub + LinkedIn" },
          { label: "Match tools", value: "Score + similarity" },
          { label: "Improvement", value: "Suggestions + survey" },
        ].map((stat) => (
          <Card key={stat.label}>
            <p className="text-xs font-medium uppercase tracking-wide text-muted">
              {stat.label}
            </p>
            <p className="mt-2 text-xl font-semibold">{stat.value}</p>
          </Card>
        ))}
      </div>

      <h3 className="mb-3 text-sm font-semibold text-foreground">
        Quick navigation
      </h3>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {quickLinks.map((item) => (
          <Link key={item.href} href={item.href}>
            <Card className="h-full transition hover:border-primary/40 hover:shadow-md">
              <p className="font-medium">{item.label}</p>
              <p className="mt-1 text-xs text-muted">{item.description}</p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
