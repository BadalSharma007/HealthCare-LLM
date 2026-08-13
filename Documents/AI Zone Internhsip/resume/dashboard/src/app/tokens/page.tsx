"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface TokenResponse {
  balance?: number;
  tokens?: number;
  message?: string;
  [key: string]: unknown;
}

interface TransactionResponse {
  transactions?: Array<{
    id?: string;
    amount?: number;
    operation?: string;
    timestamp?: string;
  }>;
  [key: string]: unknown;
}

export default function TokensPage() {
  const [balance, setBalance] = useState<number | null>(null);
  const [transactions, setTransactions] = useState<TransactionResponse | null>(null);
  const [amount, setAmount] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFetchBalance = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<TokenResponse>(
        endpoints.tokens.fetch(DEMO_USER),
        { method: "GET" }
      );
      setBalance(data.balance || data.tokens || 0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch balance");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadTransactions = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<TransactionResponse>(
        endpoints.data.transactions(DEMO_USER),
        { method: "GET" }
      );
      setTransactions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load transactions");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTokens = async () => {
    if (!amount || parseInt(amount) < 0) {
      setError("Please enter a valid amount");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const url = `${endpoints.tokens.create(DEMO_USER)}?token=${parseInt(amount)}`;
      const data = await apiFetch<TokenResponse>(url, {
        method: "POST",
      });
      setBalance(data.balance || data.tokens || 0);
      setAmount("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create tokens");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Tokens"
        description="Create or inspect token balance and transaction history that gates paid actions."
        apiHint="POST /tokens/create/{user_id} · GET /tokens/fetch/{user_id} · GET /data/transactions/{user_id}"
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            Current balance
          </p>
          <p className="mt-2 text-3xl font-semibold">
            {balance !== null ? balance : "—"}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleFetchBalance}
              disabled={loading}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
            >
              {loading ? "Fetching…" : "Fetch balance"}
            </button>
            <button
              type="button"
              onClick={handleLoadTransactions}
              disabled={loading}
              className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
            >
              {loading ? "Loading…" : "Load transactions"}
            </button>
          </div>
          {transactions && (
            <pre className="mt-4 max-h-40 overflow-auto rounded-xl bg-slate-950 p-3 text-xs text-slate-200">
              {JSON.stringify(transactions, null, 2)}
            </pre>
          )}
        </Card>
        <Card>
          <h3 className="font-semibold">Set / create tokens</h3>
          <label className="mt-3 block text-sm">
            <span className="mb-1.5 block font-medium">Amount</span>
            <input
              type="number"
              min={0}
              placeholder="100"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
            />
          </label>
          <button
            type="button"
            onClick={handleCreateTokens}
            disabled={loading}
            className="mt-4 rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
          >
            {loading ? "Creating…" : "Create / set tokens"}
          </button>
          {error && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {error}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
