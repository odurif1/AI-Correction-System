"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { api } from "@/lib/api";
import { Download, Loader2, FileJson, FileSpreadsheet, FileText } from "lucide-react";

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
  const [exporting, setExporting] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const handleExport = async (format: "csv" | "json" | "excel") => {
    setExporting(format);
    setProgress(0);

    try {
      // Simulate progress since fetch doesn't provide it
      const progressInterval = setInterval(() => {
        setProgress((p) => Math.min(p + 10, 90));
      }, 100);

      const blob = await api.exportSession(sessionId, format);

      clearInterval(progressInterval);
      setProgress(100);

      // Download the file
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      // Use appropriate file extension
      const extension = format === "excel" ? "xlsx" : format;
      a.download = `session_${sessionId}.${extension}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success(`Export ${format.toUpperCase()} téléchargé`);
    } catch (error) {
      toast.error(`Erreur export: ${(error as Error).message}`);
    } finally {
      setTimeout(() => {
        setExporting(null);
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

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant={variant} size={size} disabled={!!exporting}>
            {exporting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <Download className="h-4 w-4 mr-2" />
                Export
              </>
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => handleExport("csv")}>
            <FileSpreadsheet className="h-4 w-4 mr-2" />
            Export CSV
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => handleExport("json")}>
            <FileJson className="h-4 w-4 mr-2" />
            Export JSON
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => handleExport("excel")}>
            <FileSpreadsheet className="h-4 w-4 mr-2" />
            Export Excel
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
