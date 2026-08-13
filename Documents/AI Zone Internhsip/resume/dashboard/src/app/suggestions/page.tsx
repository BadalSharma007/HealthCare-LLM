import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function SuggestionsPage() {
  return (
    <div>
      <PageHeader
        title="Resume suggestions interview"
        description="Start an interactive Q&A that produces resume bullet-point improvements."
        apiHint="POST /suggestion/start/{user_id} · POST /suggestion/submit/{user_id} · GET /data/suggestions/{user_id}"
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">Job description</span>
            <textarea
              rows={6}
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <button
            type="button"
            className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
          >
            Start interview
          </button>
          <div className="mt-4 space-y-3">
            <p className="text-sm font-medium">Questions</p>
            <div className="rounded-xl bg-background p-3 text-sm text-muted">
              Questions from the API will appear here.
            </div>
            <label className="block text-sm">
              <span className="mb-1.5 block font-medium">Your answers</span>
              <textarea
                rows={5}
                placeholder="One answer per line…"
                className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
              />
            </label>
            <button
              type="button"
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background"
            >
              Submit answers
            </button>
          </div>
        </Card>
        <Card>
          <h3 className="font-semibold">Suggestions output</h3>
          <pre className="mt-3 max-h-[28rem] overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {`// Bullet improvements`}
          </pre>
        </Card>
      </div>
    </div>
  );
}
