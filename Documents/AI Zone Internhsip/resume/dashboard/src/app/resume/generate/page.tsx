import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function ResumeGeneratePage() {
  return (
    <div>
      <PageHeader
        title="Generate CV"
        description="Generate a formatted resume / CV from profile data and a job description."
        apiHint="POST /resume/generate/{user_id}"
      />
      <Card>
        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">Target job description</span>
          <textarea
            rows={8}
            placeholder="Paste the JD to tailor generation…"
            className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
          />
        </label>
        <button
          type="button"
          className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
        >
          Generate resume
        </button>
      </Card>
    </div>
  );
}
