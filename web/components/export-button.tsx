"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { Download, Loader2 } from "lucide-react";

interface ExportButtonProps {
  sessionId: string;
  variant?: "default" | "outline" | "ghost";
  size?: "default" | "sm" | "lg";
}

export function ExportButton({
  sessionId,
  variant = "outline",
  size = "default",
}: ExportButtonProps) {
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleExport = async () => {
    setExporting(true);
    setProgress(0);

    try {
      // Simulate progress since fetch doesn't provide it
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 100);

      const blob = await api.exportSession(sessionId, "csv");

      clearInterval(progressInterval);
      setProgress(100);

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `session_${sessionId}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export error:", error);
    } finally {
      setTimeout(() => {
        setExporting(false);
        setProgress(0);
      }, 500);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {exporting && (
        <div className="flex items-center gap-2 w-32">
          <Progress value={progress} className="h-2" />
          <span className="text-xs text-muted-foreground">{progress}%</span>
        </div>
      )}

      <Button variant={variant} size={size} onClick={handleExport} disabled={exporting}>
        {exporting ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            Export...
          </>
        ) : (
          <>
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </>
        )}
      </Button>
    </div>
  );
}
