"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface HistoryItem {
  id?: string;
  module?: string;
  action?: string;
  data?: Record<string, unknown>;
  timestamp?: string;
}

interface HistoryResponse {
  history?: HistoryItem[];
  items?: HistoryItem[];
  [key: string]: unknown;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLoadHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<HistoryResponse>(
        endpoints.user.history(DEMO_USER),
        { method: "GET" }
      );
      setHistory(data.history || data.items || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Action history"
        description="Read back actions previously generated or logged for a user."
        apiHint="GET /history/{user_id}"
      />
      <Card>
        <button
          type="button"
          onClick={handleLoadHistory}
          disabled={loading}
          className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
        >
          {loading ? "Loading…" : "Load history"}
        </button>
        {error && (
          <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
            {error}
          </div>
        )}
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-xs uppercase text-muted">
              <tr>
                <th className="py-2 pr-4">Module</th>
                <th className="py-2 pr-4">Action</th>
                <th className="py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              {history.length === 0 ? (
                <tr>
                  <td className="py-3 pr-4" colSpan={3}>
                    No history loaded yet.
                  </td>
                </tr>
              ) : (
                history.map((item, idx) => (
                  <tr key={item.id || idx} className="border-b border-border/50 hover:bg-background/50">
                    <td className="py-3 pr-4">{item.module || "—"}</td>
                    <td className="py-3 pr-4">{item.action || "—"}</td>
                    <td className="py-3 pr-4 text-xs">{item.timestamp || "—"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
