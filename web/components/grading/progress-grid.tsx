"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { CopyProgress } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  Loader2,
  AlertCircle,
  Circle,
  Clock,
} from "lucide-react";

interface ProgressGridProps {
  copies: CopyProgress[];
  totalCopies: number;
}

export function ProgressGrid({ copies, totalCopies }: ProgressGridProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
      {Array.from({ length: totalCopies }).map((_, index) => {
        const copy = copies.find((c) => c.copyIndex === index + 1);
        const status = copy?.status || "pending";

        return (
          <Card
            key={index}
            className={cn(
              "transition-all duration-200 copy-card-anim",
              "hover:shadow-md",
              status === "done" && "border-success/50 bg-success/5",
              status === "error" && "border-destructive/50 bg-destructive/5",
              status === "grading" && "border-primary/50 bg-primary/5"
            )}
          >
            <CardContent className="p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium truncate pr-2" title={copy?.studentName}>
                  {copy?.studentName || `Copy ${index + 1}`}
                </span>
                <StatusIcon status={status} />
              </div>

              {status === "grading" && copy?.questions && (
                <div className="space-y-1">
                  <Progress
                    value={
                      (copy.questions.filter((q) => q.status === "done").length /
                        copy.questions.length) *
                      100
                    }
                    className="h-1.5"
                  />
                  <p className="text-xs text-muted-foreground">
                    {copy.questions.filter((q) => q.status === "done").length}/
                    {copy.questions.length} questions
                  </p>
                </div>
              )}

              {status === "done" && copy?.totalScore !== undefined && (
                <div className="text-sm">
                  <span className="font-semibold">{copy.totalScore.toFixed(1)}</span>
                  <span className="text-muted-foreground">/{copy.maxScore}</span>
                </div>
              )}

              {status === "error" && copy?.error && (
                <p className="text-xs text-destructive line-clamp-2" title={copy.error}>
                  {copy.error}
                </p>
              )}

              {/* Show agreement status for dual-LLM grading */}
              {status === "done" && copy?.agreement !== undefined && (
                <div className="mt-1 flex items-center gap-1">
                  {copy.agreement ? (
                    <CheckCircle2 className="h-3 w-3 text-success" />
                  ) : (
                    <AlertCircle className="h-3 w-3 text-warning" />
                  )}
                  <span className="text-xs text-muted-foreground">
                    {copy.agreement ? "Accord" : "Désaccord"}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "done":
      return <CheckCircle2 className="h-4 w-4 text-success" />;
    case "grading":
      return <Loader2 className="h-4 w-4 text-primary animate-spin" />;
    case "error":
      return <AlertCircle className="h-4 w-4 text-destructive" />;
    case "pending":
      return <Clock className="h-4 w-4 text-muted-foreground" />;
    default:
      return <Circle className="h-4 w-4 text-muted-foreground" />;
  }
}
