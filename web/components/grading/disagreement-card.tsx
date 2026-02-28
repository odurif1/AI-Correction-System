"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LLMGradeCard } from "./llm-grade-card";
import type { Disagreement, ResolveDecision } from "@/lib/types";
import { AlertTriangle, Minus } from "lucide-react";

interface DisagreementCardProps {
  disagreement: Disagreement;
  onResolve: (decision: ResolveDecision) => Promise<void>;
}

export function DisagreementCard({
  disagreement,
  onResolve,
}: DisagreementCardProps) {
  const [selected, setSelected] = useState<"llm1" | "llm2" | "average" | "custom" | null>(null);
  const [customGrade, setCustomGrade] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const averageGrade =
    (disagreement.llm1.grade + disagreement.llm2.grade) / 2;

  const handleResolve = async () => {
    if (!selected) return;

    setLoading(true);
    try {
      await onResolve({
        action: selected,
        custom_grade:
          selected === "custom" ? parseFloat(customGrade) : undefined,
      });
    } catch (error) {
      console.error("Erreur lors de la résolution:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="border-warning/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-warning" />
            <Badge variant="warning">Désaccord</Badge>
          </div>
          <div className="text-sm text-muted-foreground">
            {disagreement.student_name || `Copy #${disagreement.copy_index}`} -{" "}
            {disagreement.question_id}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <LLMGradeCard
            provider={disagreement.llm1.provider}
            model={disagreement.llm1.model}
            grade={disagreement.llm1.grade}
            maxGrade={disagreement.max_points}
            confidence={disagreement.llm1.confidence}
            reasoning={disagreement.llm1.reasoning}
            reading={disagreement.llm1.reading}
            selected={selected === "llm1"}
            onClick={() => setSelected("llm1")}
            index={0}
          />
          <LLMGradeCard
            provider={disagreement.llm2.provider}
            model={disagreement.llm2.model}
            grade={disagreement.llm2.grade}
            maxGrade={disagreement.max_points}
            confidence={disagreement.llm2.confidence}
            reasoning={disagreement.llm2.reasoning}
            reading={disagreement.llm2.reading}
            selected={selected === "llm2"}
            onClick={() => setSelected("llm2")}
            index={1}
          />
        </div>

        {/* Custom grade input */}
        {selected === "custom" && (
          <div className="mt-4 space-y-2">
            <Label htmlFor="custom-grade">Note personnalisée</Label>
            <Input
              id="custom-grade"
              type="number"
              min={0}
              max={disagreement.max_points}
              step={0.5}
              value={customGrade}
              onChange={(e) => setCustomGrade(e.target.value)}
              placeholder={`Entrer la note (0-${disagreement.max_points})`}
            />
          </div>
        )}
      </CardContent>

      <CardFooter className="flex-wrap gap-2 justify-center">
        <Button
          variant={selected === "llm1" ? "default" : "outline"}
          size="sm"
          onClick={() => {
            setSelected("llm1");
          }}
        >
          Garder {disagreement.llm1.grade.toFixed(1)}
        </Button>
        <Button
          variant={selected === "llm2" ? "default" : "outline"}
          size="sm"
          onClick={() => {
            setSelected("llm2");
          }}
        >
          Garder {disagreement.llm2.grade.toFixed(1)}
        </Button>
        <Button
          variant={selected === "average" ? "default" : "outline"}
          size="sm"
          onClick={() => {
            setSelected("average");
          }}
        >
          <Minus className="h-4 w-4 mr-1" />
          Moyenne ({averageGrade.toFixed(1)})
        </Button>
        <Button
          variant={selected === "custom" ? "default" : "outline"}
          size="sm"
          onClick={() => {
            setSelected("custom");
          }}
        >
          Personnalisée
        </Button>
        <Button
          variant="default"
          size="sm"
          disabled={!selected || loading || (selected === "custom" && !customGrade)}
          onClick={handleResolve}
        >
          {loading ? "Résolution..." : "Résoudre"}
        </Button>
      </CardFooter>
    </Card>
  );
}
