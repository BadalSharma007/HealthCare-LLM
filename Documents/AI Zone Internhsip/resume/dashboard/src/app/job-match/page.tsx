"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface JobMatchResult {
  match_score?: number;
  resume_rating?: number;
  alignment?: Record<string, unknown>;
  summary?: string;
  [key: string]: unknown;
}

export default function JobMatchPage() {
  const [jd, setJd] = useState("");
  const [result, setResult] = useState<JobMatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFullMatch = async () => {
    if (!jd.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<JobMatchResult>(
        endpoints.job.match(DEMO_USER),
        {
          method: "POST",
          body: jd,
        }
      );
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to match job");
    } finally {
      setLoading(false);
    }
  };

  const handleQuickRating = async () => {
    if (!jd.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<JobMatchResult>(endpoints.job.rating, {
        method: "POST",
        body: jd,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rate resume");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Job matching"
        description="Compare the candidate resume against a job description and score alignment."
        apiHint="POST /job/match/{user_id} · POST /resume-rating · GET /data/resume-rating/{user_id}"
      />
      <Card>
        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">Job description</span>
          <textarea
            rows={10}
            placeholder="Paste JD…"
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
          />
        </label>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handleFullMatch}
            disabled={loading}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
          >
            {loading ? "Matching…" : "Full job match"}
          </button>
          <button
            type="button"
            onClick={handleQuickRating}
            disabled={loading}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
          >
            {loading ? "Rating…" : "Quick resume rating"}
          </button>
        </div>
        {error && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
            {error}
          </div>
        )}
        {result && (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
        {!result && !error && (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {`// Match score & breakdown will appear here`}
          </pre>
        )}
      </Card>
    </div>
  );
}
