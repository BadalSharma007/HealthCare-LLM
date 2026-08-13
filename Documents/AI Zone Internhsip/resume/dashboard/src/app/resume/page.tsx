"use client";

import { useRef, useState } from "react";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { apiFetch, endpoints } from "@/lib/api";

const DEMO_USER = "demo-user";

interface FetchResponse {
  resume_url?: string;
  s3_key?: string;
  message?: string;
  [key: string]: unknown;
}

export default function ResumePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [resumeUrl, setResumeUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const handleUpload = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setUploadError("Please select a PDF file");
      return;
    }

    if (file.type !== "application/pdf") {
      setUploadError("Only PDF files are accepted");
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append("pdf_file", file);

      const response = await apiFetch<FetchResponse>(
        endpoints.resume.upload(DEMO_USER),
        {
          method: "POST",
          body: formData,
        }
      );

      setResumeUrl(response.resume_url || response.s3_key || null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Failed to upload resume");
    } finally {
      setUploading(false);
    }
  };

  const handleFetch = async () => {
    setFetchLoading(true);
    setFetchError(null);

    try {
      const response = await apiFetch<FetchResponse>(
        endpoints.resume.fetch(DEMO_USER),
        {
          method: "GET",
        }
      );

      setResumeUrl(response.resume_url || response.s3_key || null);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "Failed to fetch resume URL");
    } finally {
      setFetchLoading(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Resume upload & fetch"
        description="Upload a PDF resume and retrieve the stored presigned URL for the current user."
        apiHint="POST /resume/upload/{user_id} · GET /resume/fetch/{user_id}"
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <h3 className="font-semibold">Upload resume PDF</h3>
          <p className="mt-1 text-sm text-muted">
            Only application/pdf is accepted. File is stored via S3 with presigned URLs.
          </p>
          <form className="mt-4 space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              className="block w-full text-sm text-muted file:mr-3 file:rounded-lg file:border-0 file:bg-primary file:px-3 file:py-2 file:text-sm file:font-medium file:text-white"
            />
            <button
              type="button"
              onClick={handleUpload}
              disabled={uploading}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:opacity-50"
            >
              {uploading ? "Uploading…" : "Upload"}
            </button>
          </form>
          {uploadError && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {uploadError}
            </div>
          )}
        </Card>

        <Card>
          <h3 className="font-semibold">Fetch stored resume</h3>
          <p className="mt-1 text-sm text-muted">
            Load the resume URL currently saved for this user.
          </p>
          <button
            type="button"
            onClick={handleFetch}
            disabled={fetchLoading}
            className="mt-4 rounded-lg border border-border px-4 py-2 text-sm font-medium hover:bg-background disabled:opacity-50"
          >
            {fetchLoading ? "Fetching…" : "Fetch resume URL"}
          </button>
          {resumeUrl ? (
            <div className="mt-4 space-y-2">
              <div className="rounded-xl bg-background p-3 font-mono text-xs break-all text-primary">
                {resumeUrl}
              </div>
              <a
                href={resumeUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block text-xs text-primary hover:underline"
              >
                Open in new tab
              </a>
            </div>
          ) : (
            <div className="mt-4 rounded-xl bg-background p-3 font-mono text-xs text-muted">
              resume_url: —
            </div>
          )}
          {fetchError && (
            <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-500">
              {fetchError}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
