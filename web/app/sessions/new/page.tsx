"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { FileUploader, FileWithProgress } from "@/components/grading/file-uploader";
import { PreAnalysisResults } from "@/components/grading/pre-analysis-results";
import { api, ApiError } from "@/lib/api";
import { Loader2, Search, FileText } from "lucide-react";
import { useRotatingMessage } from "@/lib/waiting-messages";
import type { PreAnalysisResult } from "@/lib/types";

type Step = "upload" | "uploading" | "analyzing" | "review" | "confirming" | "grading";

export default function NewSessionPage() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [filesWithProgress, setFilesWithProgress] = useState<FileWithProgress[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [step, setStep] = useState<Step>("upload");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [preAnalysisResult, setPreAnalysisResult] =
    useState<PreAnalysisResult | null>(null);
  const [isConfirming, setIsConfirming] = useState(false);

  // Rotating waiting message for grading step
  const waitingMessage = useRotatingMessage();

  // Polling ref for cleanup
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

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
        toast.warning(
          `${result.uploaded_count}/${files.length} fichier(s) uploadé(s)`
        );
      } else {
        toast.success(
          `${files.length} fichier${files.length > 1 ? "s" : ""} uploadé${
            files.length > 1 ? "s" : ""
          }`
        );
      }

      // Move to diagnostic step
      setStep("analyzing");

      // 3. Automatically start diagnostic
      const analysisResult = await api.preAnalyze(newSessionId);
      setPreAnalysisResult(analysisResult);
      setStep("review");
    } catch (error) {
      setStep("upload");
      setUploadProgress(0);
      console.error("Upload error:", error);
      if (error instanceof ApiError) {
        toast.error(`Erreur (${error.status}): ${error.message}`);
      } else if (error instanceof Error) {
        toast.error(`Erreur: ${error.message}`);
      } else {
        toast.error("Une erreur est survenue");
      }
    }
  };

  const handleAnalyze = async () => {
    if (!sessionId) return;

    setStep("analyzing");

    try {
      const result = await api.preAnalyze(sessionId, { force_refresh: true });
      setPreAnalysisResult(result);
      setStep("review");
    } catch (error) {
      console.error("Diagnostic error:", error);
      setStep("review"); // Go back to review even on error
      if (error instanceof ApiError) {
        toast.error(`Erreur de diagnostic (${error.status}): ${error.message}`);
      } else if (error instanceof Error) {
        toast.error(`Erreur de diagnostic: ${error.message}`);
      } else {
        toast.error("Erreur lors du diagnostic");
      }
    }
  };

  const handleConfirm = async (adjustments?: {
    grading_scale?: Record<string, number>;
  }) => {
    if (!sessionId) {
      console.error("No session ID");
      toast.error("Erreur: aucune session active");
      return;
    }

    console.log("Confirming pre-analysis for session:", sessionId);
    setIsConfirming(true);
    setStep("confirming");

    try {
      const result = await api.confirmPreAnalysis(sessionId, {
        confirm: true,
        adjustments,
      });
      console.log("Confirm result:", result);

      toast.success("Diagnostic confirmé - correction en cours");

      // Move to grading step - will poll until complete
      setStep("grading");

      // Start polling for session status
      const pollSessionStatus = async () => {
        try {
          const session = await api.getSession(sessionId);
          console.log("Session status:", session.status);

          if (session.status === "complete") {
            // Stop polling
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            // Redirect to review tab
            router.push(`/sessions/${sessionId}?tab=review`);
          } else if (session.status === "error") {
            // Stop polling on error
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            toast.error("Une erreur est survenue pendant la correction");
            setStep("review");
            setIsConfirming(false);
          }
        } catch (pollError) {
          console.error("Poll error:", pollError);
          // Don't stop polling on temporary errors, keep trying
        }
      };

      // Poll every 2 seconds
      pollingRef.current = setInterval(pollSessionStatus, 2000);

      // Also poll immediately
      pollSessionStatus();

    } catch (error) {
      console.error("Confirm error:", error);
      setIsConfirming(false);
      setStep("review");
      if (error instanceof ApiError) {
        console.error("API Error details:", error.status, error.message);
        toast.error(`Erreur (${error.status}): ${error.message}`);
      } else if (error instanceof Error) {
        toast.error(`Erreur: ${error.message}`);
      } else {
        toast.error("Erreur lors de la confirmation");
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8 max-w-3xl">
        <div className="space-y-6">
          <div className="flex items-center justify-center">
            <div className="text-center">
              <p className="text-muted-foreground text-sm">
                {step === "upload" || step === "uploading"
                  ? "Glissez vos copies PDF pour commencer"
                  : step === "analyzing"
                  ? "Diagnostic du document en cours..."
                  : step === "grading"
                  ? waitingMessage
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

          {/* Analyzing Step */}
          {step === "analyzing" && (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div className="text-center">
                <h2 className="text-lg font-medium">Diagnostic du document</h2>
                <p className="text-muted-foreground text-sm">
                  Détection de la structure et du barème...
                </p>
              </div>
            </div>
          )}

          {/* Review Step */}
          {step === "review" && preAnalysisResult && (
            <>
              {/* File info - now shows multiple files or single file */}
              {files.length === 1 ? (
                <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                  <FileText className="h-5 w-5 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">
                      {files[0]?.name || "Document PDF"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {preAnalysisResult.page_count} pages
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleAnalyze}
                    disabled={step !== "review"}
                  >
                    <Search className="h-4 w-4 mr-1" />
                    Rediagnostiquer
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2 p-3 bg-muted rounded-lg">
                  <FileText className="h-5 w-5 text-muted-foreground" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">
                      {files.length} documents PDF
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {preAnalysisResult.page_count} pages totales
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleAnalyze}
                    disabled={step !== "review"}
                  >
                    <Search className="h-4 w-4 mr-1" />
                    Rediagnostiquer
                  </Button>
                </div>
              )}

              {/* Analysis Results */}
              <PreAnalysisResults
                result={preAnalysisResult}
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

          {/* Grading Step - stays until complete */}
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
      </main>

      <Footer />
    </div>
  );
}
