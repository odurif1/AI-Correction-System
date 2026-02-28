"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { LLMGrade } from "@/lib/types";
import { cn } from "@/lib/utils";

interface LLMGradeCardProps {
  provider: string;
  model?: string;
  grade: number;
  maxGrade: number;
  reasoning?: string;
  confidence?: number;
  reading?: string;
  selected?: boolean;
  onClick?: () => void;
  index?: number; // 0 = A, 1 = B
}

export function LLMGradeCard({
  grade,
  maxGrade,
  reasoning,
  reading,
  selected,
  onClick,
  index = 0,
}: LLMGradeCardProps) {
  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        selected && "ring-2 ring-primary"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Intelligence artificielle - {index === 0 ? 'A' : 'B'}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Grade display */}
        <div className="text-center">
          <div className="text-3xl font-bold">
            {grade.toFixed(1)}
            <span className="text-lg text-muted-foreground">/{maxGrade}</span>
          </div>
        </div>

        {/* Reading */}
        {reading && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">
              Réponse élève
            </p>
            <p className="text-sm bg-muted/50 rounded p-2">{reading}</p>
          </div>
        )}

        {/* Reasoning */}
        {reasoning && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">
              Explication
            </p>
            <p className="text-sm text-muted-foreground">
              {reasoning}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
