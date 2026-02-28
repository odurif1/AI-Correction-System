"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
}

export function LLMGradeCard({
  provider,
  model,
  grade,
  maxGrade,
  reasoning,
  reading,
  selected,
  onClick,
}: LLMGradeCardProps) {
  const providerColors: Record<string, string> = {
    gemini: "bg-blue-500/10 text-blue-500 border-blue-500/20",
    openai: "bg-green-500/10 text-green-500 border-green-500/20",
    openrouter: "bg-purple-500/10 text-purple-500 border-purple-500/20",
  };

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        selected && "ring-2 ring-primary"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg">{provider}</CardTitle>
            {model && (
              <p className="text-xs text-muted-foreground">{model}</p>
            )}
          </div>
          <Badge
            variant="outline"
            className={providerColors[provider.toLowerCase()] || ""}
          >
            {provider}
          </Badge>
        </div>
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
            <p className="text-sm text-muted-foreground line-clamp-3">
              {reasoning}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
