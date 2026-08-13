"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface SimilarityResult {
  overall_score?: number;
  work_score?: number;
  breakdown?: Record<string, unknown>;
  [key: string]: unknown;
}

export default function SimilarityPage() {
  const [jd, setJd] = useState("");
  const [result, setResult] = useState<SimilarityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleOverallSimilarity = async () => {
    if (!jd.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<SimilarityResult>(
        endpoints.similarity.overall(DEMO_USER),
        {
          method: "POST",
          body: jd,
        }
      );
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze overall similarity");
    } finally {
      setLoading(false);
    }
  };

  const handleWorkSimilarity = async () => {
    if (!jd.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<SimilarityResult>(
        endpoints.similarity.work(DEMO_USER),
        {
          method: "POST",
          body: jd,
        }
      );
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze work similarity");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Work / project similarity"
        description="Deep breakdown of work experience and project alignment against a JD."
        apiHint="POST /similarity/overall/{user_id} · POST /similarity/work/{user_id} · GET /data/similarity/{user_id}"
      />
      <Card>
        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">Job description</span>
          <textarea
            rows={8}
            placeholder="Paste JD…"
            value={jd}
            onChange={(e) => setJd(e.target.value)}
            className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
          />
        </label>
        <div className="mt-4 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handleOverallSimilarity}
            disabled={loading}
            className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "Overall similarity"}
          </button>
          <button
            type="button"
            onClick={handleWorkSimilarity}
            disabled={loading}
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
          >
            {loading ? "Analyzing…" : "Work-only similarity"}
          </button>
        </div>
        {error && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
            {error}
          </div>
        )}
        {result ? (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {`// Similarity payload will appear here`}
          </pre>
        )}
      </Card>
    </div>
  );
}
