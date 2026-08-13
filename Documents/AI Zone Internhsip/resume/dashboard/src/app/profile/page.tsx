import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function JobProfilePage() {
  return (
    <div>
      <PageHeader
        title="Job profile"
        description="Aggregate resume + GitHub + LinkedIn + applied jobs into one candidate profile."
        apiHint="POST /job/profile/update/{user_id} · GET /data/job-profile/{user_id}"
      />
      <Card>
        <form className="grid gap-4 sm:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">GitHub URL</span>
            <input
              type="url"
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">LinkedIn URL</span>
            <input
              type="url"
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">Personal site</span>
            <input
              type="url"
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <label className="block text-sm sm:col-span-2">
            <span className="mb-1.5 block font-medium">
              Applied jobs (one per line)
            </span>
            <textarea
              rows={4}
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
        </form>
        <button
          type="button"
          className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
        >
          Update profile
        </button>
      </Card>
    </div>
  );
}
