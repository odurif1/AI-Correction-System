"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { FileUploader, FileWithProgress } from "@/components/grading/file-uploader";
import { DetectionResults } from "@/components/grading/detection-results";
import { api, ApiError } from "@/lib/api";
import { Loader2, Search, FileText } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { useProgressSocket } from "@/lib/websocket";
import { useRotatingMessage } from "@/lib/waiting-messages";
import type { DetectionResult } from "@/lib/types";

type Step = "upload" | "uploading" | "detecting" | "review" | "confirming" | "grading";

function NewSessionContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user } = useAuth();
  const [files, setFiles] = useState<File[]>([]);
  const [filesWithProgress, setFilesWithProgress] = useState<FileWithProgress[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [step, setStep] = useState<Step>("upload");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] =
    useState<DetectionResult | null>(null);
  const [isConfirming, setIsConfirming] = useState(false);

  // WebSocket connection for grading progress (only when grading)
  const { subPhase, subPhaseLabel } = useProgressSocket({
    sessionId: sessionId || "",
    onEvent: (event) => {
      if (event.type === "session_complete") {
        router.push(`/sessions/${sessionId}?tab=review`);
      }
    },
    onError: (error) => {
      console.error("WebSocket error:", error.message);
      setStep("review");
      setIsConfirming(false);
    },
  });

  // Rotating waiting message (20 second interval)
  const waitingMessage = useRotatingMessage();

  // Polling ref for cleanup
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Handle resume from URL parameter
  useEffect(() => {
    const resumeSessionId = searchParams.get("resume");
    if (resumeSessionId) {
      resumeSession(resumeSessionId);
    }
  }, [searchParams]);

  const resumeSession = async (resumeSessionId: string) => {
    setStep("detecting");
    setSessionId(resumeSessionId);

    try {
      // Get session to check status
      const session = await api.getSession(resumeSessionId);

      if (session.status === "diagnostic") {
        // Try to load existing detection (cached)
        const result = await api.detect(resumeSessionId, { force_refresh: false });
        setDetectionResult(result);
        setStep("review");
      } else if (session.status === "correction") {
        // Already grading, show grading screen
        setStep("grading");
      } else if (session.status === "complete") {
        // Already complete, redirect to review
        router.push(`/sessions/${resumeSessionId}?tab=review`);
      } else if (session.status === "error") {
        // Error state, allow re-upload
        setStep("upload");
      }
    } catch (error) {
      console.error("Resume error:", error);
      setStep("upload");
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // Handle files selected from FileUploader
  const handleFilesSelected = (newFiles: File[]) => {
    setFiles(newFiles);
    // Initialize progress for all files
    setFilesWithProgress(newFiles.map(file => ({ file, progress: 0 })));
  };

  // Update progress for a specific file
  const setFileProgress = (fileIndex: number, progress: number) => {
    setFilesWithProgress(prev => {
      const updated = [...prev];
      if (updated[fileIndex]) {
        updated[fileIndex] = { ...updated[fileIndex], progress };
      }
      return updated;
    });
  };

  // Set error for a specific file
  const setFileError = (fileIndex: number, error: string) => {
    setFilesWithProgress(prev => {
      const updated = [...prev];
      if (updated[fileIndex]) {
        updated[fileIndex] = { ...updated[fileIndex], error };
      }
      return updated;
    });
  };

  const handleStartUpload = async () => {
    if (files.length === 0) return;
    setStep("uploading");
    setUploadProgress(0);

    try {
      // 1. Create the session (empty)
      const session = await api.createSession({});
      const newSessionId = session.session_id;
      setSessionId(newSessionId);

      // 2. Upload the files with per-file progress
      const result = await api.uploadPdfs(newSessionId, files, (fileIndex, progress) => {
        // Update progress for specific file
        setFileProgress(fileIndex, progress);
        // Also update overall progress
        setUploadProgress(Math.round(((fileIndex + 1) / files.length) * progress));
      });

      // Handle any validation errors from response
      if (result.errors && result.errors.length > 0) {
        result.errors.forEach(({ index, error }) => {
          setFileError(index, error);
        });
      }

      // Move to detection step
      setStep("detecting");

      // 3. Automatically start detection
      const detectResult = await api.detect(newSessionId);
      setDetectionResult(detectResult);
      setStep("review");
    } catch (error) {
      setStep("upload");
      setUploadProgress(0);
      console.error("Upload error:", error);
    }
  };

  const handleDetect = async () => {
    if (!sessionId) return;

    setStep("detecting");

    try {
      const result = await api.detect(sessionId, { force_refresh: true });
      setDetectionResult(result);
      setStep("review");
    } catch (error) {
      console.error("Detection error:", error);
      setStep("review"); // Go back to review even on error
    }
  };

  const handleConfirm = async (adjustments?: {
    grading_scale?: Record<string, number>;
  }) => {
    if (!sessionId) {
      console.error("No session ID");
      return;
    }

    console.log("Confirming detection for session:", sessionId);
    setIsConfirming(true);
    setStep("confirming");

    try {
      const result = await api.confirmDetection(sessionId, {
        confirm: true,
        adjustments,
      });
      console.log("Confirm result:", result);

      // Move to grading step - WebSocket will track sub-phases
      setStep("grading");
      setIsConfirming(false);

    } catch (error) {
      console.error("Confirm error:", error);
      setIsConfirming(false);
      setStep("review");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground text-sm">
            {step === "upload" || step === "uploading"
              ? "Glissez vos copies PDF pour commencer"
              : step === "detecting"
              ? "Diagnostic du document en cours..."
              : ""}
          </p>
        </div>
      </div>

      {/* Upload Step */}
      {(step === "upload" || step === "uploading") && (
        <>
          <FileUploader
            onFilesSelected={handleFilesSelected}
            multiple={true}
            maxFiles={50}
            files={filesWithProgress}
            uploading={step === "uploading"}
          />

          {step === "uploading" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">
                  Chargement des fichiers...
                </span>
                <span className="font-medium">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}

          <div className="flex justify-center pt-2">
            <Button
              size="lg"
              onClick={handleStartUpload}
              disabled={files.length === 0 || step === "uploading"}
              className="min-w-[200px] bg-purple-600 hover:bg-purple-700 text-white disabled:opacity-100 disabled:bg-purple-600/50 transition-colors"
            >
              {step === "uploading" ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Chargement...
                </>
              ) : (
                "Corriger les copies"
              )}
            </Button>
          </div>
        </>
      )}

      {/* Detecting Step */}
      {step === "detecting" && (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <div className="text-center">
            <h2 className="text-lg font-medium">Détection du document</h2>
            <p className="text-muted-foreground text-sm">
              Détection de la structure et du barème...
            </p>
          </div>
        </div>
      )}

      {/* Review Step */}
      {step === "review" && detectionResult && (
        <>
          {/* Exam name header */}
          {detectionResult.exam_name && (
            <div className="text-center mb-4">
              <h2 className="text-2xl font-bold">{detectionResult.exam_name}</h2>
            </div>
          )}

          {/* File info - now shows multiple files or single file */}
          {files.length === 1 ? (
            <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
              <FileText className="h-5 w-5 text-muted-foreground" />
              <div className="flex-1">
                <p className="font-medium text-sm">
                  {files[0]?.name || "Document PDF"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {detectionResult.page_count} pages
                </p>
              </div>
              {user?.subscription_tier !== "free" && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDetect}
                  disabled={step !== "review"}
                >
                  <Search className="h-4 w-4 mr-1" />
                  Redétecter
                </Button>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
              <FileText className="h-5 w-5 text-muted-foreground" />
              <div className="flex-1">
                <p className="font-medium text-sm">
                  {files.length} documents PDF
                </p>
                <p className="text-xs text-muted-foreground">
                  {detectionResult.page_count} pages totales
                </p>
              </div>
              {user?.subscription_tier !== "free" && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDetect}
                  disabled={step !== "review"}
                >
                  <Search className="h-4 w-4 mr-1" />
                  Redétecter
                </Button>
              )}
            </div>
          )}

          {/* Detection Results */}
          <DetectionResults
            result={detectionResult}
            onConfirm={handleConfirm}
            isConfirming={isConfirming}
          />
        </>
      )}

      {/* Confirming Step */}
      {step === "confirming" && (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <div className="text-center">
            <h2 className="text-lg font-medium">Préparation de la correction</h2>
            <p className="text-muted-foreground text-sm">
              Configuration en cours...
            </p>
          </div>
        </div>
      )}

      {/* Grading Step - shows rotating waiting messages */}
      {step === "grading" && (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <Loader2 className="h-12 w-12 animate-spin text-purple-600" />
          <div className="text-center">
            <h2 className="text-lg font-medium">Correction en cours</h2>
            <p className="text-muted-foreground text-sm mt-2">
              {waitingMessage}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function LoadingFallback() {
  return (
    <div className="flex flex-col items-center justify-center py-12 space-y-4">
      <Loader2 className="h-12 w-12 animate-spin text-primary" />
      <div className="text-center">
        <h2 className="text-lg font-medium">Chargement...</h2>
      </div>
    </div>
  );
}

export default function NewSessionPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8 max-w-3xl">
        <Suspense fallback={<LoadingFallback />}>
          <NewSessionContent />
        </Suspense>
      </main>

      <Footer />
    </div>
  );
}
