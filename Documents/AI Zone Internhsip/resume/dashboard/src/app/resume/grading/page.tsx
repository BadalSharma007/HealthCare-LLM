import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function ResumeGradingPage() {
  return (
    <div>
      <PageHeader
        title="Resume structure grading"
        description="Grade resume structure and metrics quality."
        apiHint="GET /resume/grading/{user_id} · GET /data/grading/{user_id}"
      />
      <Card>
        <button
          type="button"
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
        >
          Run grading
        </button>
        <pre className="mt-4 max-h-96 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
          {`// Grading scores will render here`}
        </pre>
      </Card>
    </div>
  );
}
