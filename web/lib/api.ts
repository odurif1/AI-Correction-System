import type {
  Session,
  SessionDetail,
  Analytics,
  Disagreement,
  Provider,
  ProgressEvent,
  ResolveDecision,
  CreateSessionRequest,
  SessionResponse,
  PreAnalysisResult,
  PreAnalysisRequest,
  ConfirmPreAnalysisRequest,
  ConfirmPreAnalysisResponse,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async fetchJson<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    // Get token from localStorage
    const token = typeof window !== 'undefined' ? localStorage.getItem("token") : null;

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "Authorization": `Bearer ${token}` } : {}),
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return response.json();
  }

  // Sessions
  async listSessions(): Promise<{ sessions: Session[] }> {
    return this.fetchJson("/api/sessions");
  }

  async getSession(sessionId: string): Promise<SessionDetail> {
    return this.fetchJson(`/api/sessions/${sessionId}`);
  }

  async createSession(data: CreateSessionRequest): Promise<SessionResponse> {
    return this.fetchJson("/api/sessions", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    return this.fetchJson(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    });
  }

  async uploadPdfs(
    sessionId: string,
    files: File[],
    onProgress?: (fileIndex: number, progress: number) => void
  ): Promise<{
    session_id: string;
    uploaded_count: number;
    paths: string[];
    errors?: Array<{ index: number; error: string }>;
  }> {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("files", file);
      });

      const xhr = new XMLHttpRequest();

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          // Update progress for all files since we're uploading in a single request
          for (let i = 0; i < files.length; i++) {
            onProgress(i, progress);
          }
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            // Mark all files as complete on success
            if (onProgress) {
              for (let i = 0; i < files.length; i++) {
                onProgress(i, 100);
              }
            }
            resolve(response);
          } catch {
            reject(new ApiError("Invalid response format", xhr.status));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new ApiError(error.detail || `Upload failed: ${xhr.status}`, xhr.status));
          } catch {
            reject(new ApiError(`Upload failed: ${xhr.status}`, xhr.status));
          }
        }
      };

      xhr.onerror = () => {
        reject(new ApiError("Network error during upload", 0));
      };

      // Add auth token
      const token = typeof window !== 'undefined' ? localStorage.getItem("token") : null;

      xhr.open("POST", `${this.baseUrl}/api/sessions/${sessionId}/upload`);

      if (token) {
        xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      }

      xhr.send(formData);
    });
  }

  async startGrading(sessionId: string): Promise<{
    success: boolean;
    session_id: string;
    graded_count: number;
    total_count: number;
    pending_review: number;
  }> {
    return this.fetchJson(`/api/sessions/${sessionId}/grade`, {
      method: "POST",
    });
  }

  async getAnalytics(sessionId: string): Promise<Analytics> {
    return this.fetchJson(`/api/sessions/${sessionId}/analytics`);
  }

  // Disagreements
  async getDisagreements(sessionId: string): Promise<Disagreement[]> {
    return this.fetchJson(`/api/sessions/${sessionId}/disagreements`);
  }

  async resolveDisagreement(
    sessionId: string,
    questionId: string,
    decision: ResolveDecision
  ): Promise<{ success: boolean }> {
    return this.fetchJson(
      `/api/sessions/${sessionId}/disagreements/${questionId}/resolve`,
      {
        method: "POST",
        body: JSON.stringify(decision),
      }
    );
  }

  // Decisions
  async submitDecision(
    sessionId: string,
    decision: {
      question_id: string;
      copy_id: string;
      teacher_guidance: string;
      original_score: number;
      new_score: number;
      applies_to_all?: boolean;
    }
  ): Promise<{ success: boolean; updated_count: number; extracted_rule: string }> {
    return this.fetchJson(`/api/sessions/${sessionId}/decisions`, {
      method: "POST",
      body: JSON.stringify(decision),
    });
  }

  // Exports
  async exportSession(
    sessionId: string,
    format: "csv" | "json" | "pdf"
  ): Promise<Blob> {
    const response = await fetch(
      `${this.baseUrl}/api/sessions/${sessionId}/export/${format}`
    );
    if (!response.ok) {
      throw new ApiError(`Export failed: ${response.status}`, response.status);
    }
    return response.blob();
  }

  // Providers
  async listProviders(): Promise<Provider[]> {
    return this.fetchJson("/api/providers");
  }

  // Settings
  async getSettings(): Promise<Record<string, unknown>> {
    return this.fetchJson("/api/settings");
  }

  async updateSettings(
    settings: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    return this.fetchJson("/api/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  }

  // WebSocket for real-time progress
  createProgressSocket(sessionId: string): WebSocket {
    const wsProtocol = this.baseUrl.startsWith("https") ? "wss" : "ws";
    const wsHost = this.baseUrl.replace(/^https?:\/\//, "");
    return new WebSocket(`${wsProtocol}://${wsHost}/api/sessions/${sessionId}/ws`);
  }

  // Pre-analysis
  async preAnalyze(sessionId: string, request: PreAnalysisRequest = {}): Promise<PreAnalysisResult> {
    return this.fetchJson(`/api/sessions/${sessionId}/pre-analyze`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  async confirmPreAnalysis(
    sessionId: string,
    request: ConfirmPreAnalysisRequest
  ): Promise<ConfirmPreAnalysisResponse> {
    return this.fetchJson(`/api/sessions/${sessionId}/confirm-pre-analysis`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }
}

export const api = new ApiClient();
export { ApiClient, ApiError };
