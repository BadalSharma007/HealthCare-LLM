import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function SettingsPage() {
  return (
    <div>
      <PageHeader
        title="Settings"
        description="View stored user details and dashboard configuration."
        apiHint="GET /user/fetch/{user_id}"
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <h3 className="font-semibold">User record</h3>
          <button
            type="button"
            className="mt-3 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
          >
            Fetch user
          </button>
          <pre className="mt-4 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {`// user details`}
          </pre>
        </Card>
        <Card>
          <h3 className="font-semibold">Environment</h3>
          <dl className="mt-3 space-y-2 text-sm">
            <div className="flex justify-between gap-4">
              <dt className="text-muted">API base</dt>
              <dd className="font-mono text-xs">
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
              </dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Landing app</dt>
              <dd className="font-mono text-xs">http://localhost:3000</dd>
            </div>
            <div className="flex justify-between gap-4">
              <dt className="text-muted">Dashboard</dt>
              <dd className="font-mono text-xs">http://localhost:3001</dd>
            </div>
          </dl>
        </Card>
      </div>
    </div>
  );
}
