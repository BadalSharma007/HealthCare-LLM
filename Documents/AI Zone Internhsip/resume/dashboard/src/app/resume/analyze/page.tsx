import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";

export default function ResumeAnalyzePage() {
  return (
    <div>
      <PageHeader
        title="Resume analysis"
        description="Break the stored resume into structured sections for downstream matching and grading."
        apiHint="POST /resume/analyze/{user_id} · GET /data/analysis/{user_id}"
      />
      <Card>
        <p className="text-sm text-muted">
          Runs structured extraction (experience, projects, skills, education,
          etc.) and logs history under the resume module.
        </p>
        <button
          type="button"
          className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover"
        >
          Analyze resume
        </button>
        <pre className="mt-4 max-h-96 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
          {`// Analysis JSON will render here`}
        </pre>
      </Card>
    </div>
  );
}
