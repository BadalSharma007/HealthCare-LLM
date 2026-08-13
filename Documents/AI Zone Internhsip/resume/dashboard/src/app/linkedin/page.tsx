"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface LinkedinResponse {
  headline?: string;
  skills?: string[];
  experience?: Array<Record<string, unknown>>;
  profile?: Record<string, unknown>;
  [key: string]: unknown;
}

export default function LinkedinPage() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState<LinkedinResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!url.trim()) {
      setError("Please enter a LinkedIn URL");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<LinkedinResponse>(
        endpoints.linkedin.analyze(DEMO_USER),
        {
          method: "POST",
          body: JSON.stringify({ linkedin_url: url }),
          headers: { "Content-Type": "application/json" },
        }
      );
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze LinkedIn profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="LinkedIn profile"
        description="Scrape structured data such as headline, skills, and connections from a LinkedIn URL."
        apiHint="POST /linkedin/analyze/{user_id}"
      />
      <Card>
        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">LinkedIn profile URL</span>
          <input
            type="url"
            placeholder="https://www.linkedin.com/in/…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
          />
        </label>
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={loading}
          className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          {loading ? "Analyzing…" : "Analyze LinkedIn"}
        </button>
        {error && (
          <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
            {error}
          </div>
        )}
        {result ? (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {JSON.stringify(result, null, 2)}
          </pre>
        ) : (
          <pre className="mt-4 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
            {`// LinkedIn structured data will appear here`}
          </pre>
        )}
      </Card>
    </div>
  );
}
