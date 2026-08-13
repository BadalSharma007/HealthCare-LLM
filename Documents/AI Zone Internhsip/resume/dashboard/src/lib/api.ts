/**
 * Thin client for the Resume-Data_Gather FastAPI backend.
 * Set NEXT_PUBLIC_API_URL in .env.local (default http://localhost:8000).
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export function getApiUrl() {
  return API_URL;
}

export async function apiFetch<T = unknown>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;

  const isFormData = init?.body instanceof FormData;
  const isPlainText = typeof init?.body === "string";
  const defaultContentType = isFormData ? undefined : isPlainText ? "text/plain" : "application/json";

  const res = await fetch(url, {
    ...init,
    headers: {
      ...(defaultContentType ? { "Content-Type": defaultContentType } : {}),
      ...init?.headers,
    },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

/** Backend route map used by dashboard pages */
export const endpoints = {
  resume: {
    upload: (userId: string) => `/resume/upload/${userId}`,
    fetch: (userId: string) => `/resume/fetch/${userId}`,
    analyze: (userId: string) => `/resume/analyze/${userId}`,
    grading: (userId: string) => `/resume/grading/${userId}`,
    generate: (userId: string) => `/resume/generate/${userId}`,
  },
  optimize: "/optimize/resume",
  github: {
    save: (userId: string) => `/github/save/${userId}`,
    fetch: (userId: string) => `/github/fetch/${userId}`,
    analyze: (userId: string) => `/github/analyze/${userId}`,
  },
  linkedin: {
    analyze: (userId: string) => `/linkedin/analyze/${userId}`,
  },
  job: {
    match: (userId: string) => `/job/match/${userId}`,
    profileUpdate: (userId: string) => `/job/profile/update/${userId}`,
    rating: "/resume-rating",
  },
  similarity: {
    overall: (userId: string) => `/similarity/overall/${userId}`,
    work: (userId: string) => `/similarity/work/${userId}`,
  },
  suggestion: {
    start: (userId: string) => `/suggestion/start/${userId}`,
    submit: (userId: string) => `/suggestion/submit/${userId}`,
  },
  survey: {
    start: (userId: string) => `/survey/start/${userId}`,
    submit: (userId: string) => `/survey/submit/${userId}`,
  },
  data: {
    suggestions: (userId: string) => `/data/suggestions/${userId}`,
    grading: (userId: string) => `/data/grading/${userId}`,
    similarity: (userId: string) => `/data/similarity/${userId}`,
    analysis: (userId: string) => `/data/analysis/${userId}`,
    jobProfile: (userId: string) => `/data/job-profile/${userId}`,
    optimizedResume: (userId: string) => `/data/optimized-resume/${userId}`,
    resumeRating: (userId: string) => `/data/resume-rating/${userId}`,
    transactions: (userId: string) => `/data/transactions/${userId}`,
  },
  user: {
    fetch: (userId: string) => `/user/fetch/${userId}`,
    history: (userId: string) => `/history/${userId}`,
  },
  tokens: {
    create: (userId: string) => `/tokens/create/${userId}`,
    fetch: (userId: string) => `/tokens/fetch/${userId}`,
  },
} as const;
