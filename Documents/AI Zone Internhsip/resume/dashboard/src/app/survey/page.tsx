"use client";

import { useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface Question {
  id: string;
  text: string;
  type?: "yes_no" | "text" | "metric";
}

interface SurveyResponse {
  stage?: number;
  questions?: Question[];
  analysis?: Record<string, unknown>;
  complete?: boolean;
  [key: string]: unknown;
}

export default function SurveyPage() {
  const [jd, setJd] = useState("");
  const [surveyStarted, setSurveyStarted] = useState(false);
  const [currentQuestions, setCurrentQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [response, setResponse] = useState<SurveyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartSurvey = async () => {
    if (!jd.trim()) {
      setError("Please enter a job description");
      return;
    }

    setLoading(true);
    setError(null);
    setAnswers({});

    try {
      const data = await apiFetch<SurveyResponse>(
        endpoints.survey.start(DEMO_USER),
        {
          method: "POST",
          body: jd,
        }
      );
      setResponse(data);
      setSurveyStarted(true);
      if (data.questions) {
        setCurrentQuestions(data.questions);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start survey");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitStage = async () => {
    if (Object.keys(answers).length === 0) {
      setError("Please answer at least one question");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await apiFetch<SurveyResponse>(
        endpoints.survey.submit(DEMO_USER),
        {
          method: "POST",
          body: JSON.stringify(answers),
          headers: { "Content-Type": "application/json" },
        }
      );
      setResponse(data);
      if (data.complete) {
        setSurveyStarted(false);
      } else if (data.questions) {
        setCurrentQuestions(data.questions);
        setAnswers({});
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit stage");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="User experience survey"
        description="Structured multi-stage survey that turns unwritten experience into resume-ready content (screening → deep dive)."
        apiHint="POST /survey/start/{user_id} · POST /survey/submit/{user_id}"
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          {!surveyStarted ? (
            <>
              <label className="block text-sm">
                <span className="mb-1.5 block font-medium">Job description</span>
                <textarea
                  rows={5}
                  placeholder="Paste JD…"
                  value={jd}
                  onChange={(e) => setJd(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none ring-primary focus:ring-2"
                />
              </label>
              <button
                type="button"
                onClick={handleStartSurvey}
                disabled={loading}
                className="mt-4 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
              >
                {loading ? "Starting…" : "Start survey"}
              </button>
            </>
          ) : (
            <>
              <div className="mb-6">
                <p className="text-sm font-medium">Current stage questions</p>
                <div className="mt-3 space-y-4">
                  {currentQuestions.map((q) => (
                    <label key={q.id} className="block text-sm">
                      <span className="mb-2 block font-medium">{q.text}</span>
                      <input
                        type={q.type === "metric" ? "number" : "text"}
                        placeholder={
                          q.type === "yes_no"
                            ? "yes / no"
                            : q.type === "metric"
                              ? "0"
                              : "Type answer…"
                        }
                        value={answers[q.id] || ""}
                        onChange={(e) =>
                          setAnswers({ ...answers, [q.id]: e.target.value })
                        }
                        className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none ring-primary focus:ring-2"
                      />
                    </label>
                  ))}
                </div>
              </div>
              <button
                type="button"
                onClick={handleSubmitStage}
                disabled={loading}
                className="rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
              >
                {loading ? "Submitting…" : "Submit stage"}
              </button>
            </>
          )}
          {error && (
            <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {error}
            </div>
          )}
        </Card>
        <Card>
          <h3 className="font-semibold">Survey progress / final analysis</h3>
          <p className="mt-1 text-xs text-muted">
            Mid-survey returns next batch; final stage returns experience
            analysis aligned to PAR/STAR and metric categories.
          </p>
          {response ? (
            <pre className="mt-3 max-h-[28rem] overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
              {JSON.stringify(response, null, 2)}
            </pre>
          ) : (
            <pre className="mt-3 max-h-[28rem] overflow-auto rounded-xl bg-slate-950 p-4 text-xs text-slate-200">
              {`// Stage response or final analysis will appear here`}
            </pre>
          )}
        </Card>
      </div>
    </div>
  );
}
