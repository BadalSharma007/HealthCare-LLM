"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface GithubResponse {
  url?: string;
  github_url?: string;
  analysis?: Record<string, unknown>;
  repositories?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export default function GithubPage() {
  const [url, setUrl] = useState("");
  const [savedUrl, setSavedUrl] = useState<string | null>(null);
  const [result, setResult] = useState<GithubResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSaveUrl = async () => {
    if (!url.trim()) {
      setError("Please enter a GitHub URL");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<GithubResponse>(
        endpoints.github.save(DEMO_USER),
        {
          method: "POST",
          body: JSON.stringify({ github_url: url }),
          headers: { "Content-Type": "application/json" },
        }
      );
      setSavedUrl(url);
      setUrl("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save URL");
    } finally {
      setLoading(false);
    }
  };

  const handleFetchSaved = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<GithubResponse>(
        endpoints.github.fetch(DEMO_USER),
        { method: "GET" }
      );
      setSavedUrl(data.url || data.github_url || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch saved URL");
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<GithubResponse>(
        endpoints.github.analyze(DEMO_USER),
        { method: "POST" }
      );
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze GitHub profile");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="GitHub profile"
        description="Save a GitHub URL, fetch it, and scrape repo-level analysis for the candidate."
        apiHint="POST /github/save/{user_id} · GET /github/fetch/{user_id} · POST /github/analyze/{user_id}"
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <label className="block text-sm">
            <span className="mb-1.5 block font-medium">GitHub profile URL</span>
            <input
              type="url"
              placeholder="https://github.com/username"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleSaveUrl}
              disabled={loading}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
            >
              {loading ? "Saving…" : "Save URL"}
            </button>
            <button
              type="button"
              onClick={handleFetchSaved}
              disabled={loading}
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
            >
              {loading ? "Fetching…" : "Fetch saved"}
            </button>
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={loading || !savedUrl}
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
            >
              {loading ? "Analyzing…" : "Analyze"}
            </button>
          </div>
          {savedUrl && (
            <div className="mt-3 rounded-xl bg-background p-3 font-mono text-xs break-all text-primary">
              {savedUrl}
            </div>
          )}
          {error && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {error}
            </div>
          )}
        </Card>
        <Card>
          <h3 className="font-semibold">Analysis output</h3>
          {result ? (
            <pre className="mt-3 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
              {JSON.stringify(result, null, 2)}
            </pre>
          ) : (
            <pre className="mt-3 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
              {`// GitHub scrape result will appear here`}
            </pre>
          )}
        </Card>
      </div>
    </div>
  );
}
