import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function ResumeOptimizePage() {
  return (
    <div>
      <PageHeader
        title="Optimize resume"
        description="Rewrite resume markdown targeted to a specific job description."
        apiHint="POST /optimize/resume · GET /data/optimized-resume/{user_id}"
      />
      <Card>
        <form className="space-y-4">
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">Job description</span>
            <textarea
              rows={8}
              placeholder="Paste the full JD…"
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">
              Resume URL (optional override)
            </span>
            <input
              type="url"
              placeholder="https://…"
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <button
            type="button"
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
          >
            Optimize
          </button>
        </form>
        <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
          {`// updated_resume_markdown`}
        </pre>
      </Card>
    </div>
  );
}
