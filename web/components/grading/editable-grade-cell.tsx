"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { RotateCcw, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EditableGradeCellProps {
  sessionId: string;
  copyId: string;
  questionId: string;
  grade: number;
  maxPoints: number;
  hasDisagreement?: boolean;
}

export function EditableGradeCell({
  sessionId,
  copyId,
  questionId,
  grade,
  maxPoints,
  hasDisagreement,
}: EditableGradeCellProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(grade);
  const [originalGrade, setOriginalGrade] = useState(grade);
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationFn: (newGrade: number) =>
      api.updateGrade(sessionId, copyId, questionId, { new_grade: newGrade }),
    onSuccess: (_, newGrade) => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      setOriginalGrade(newGrade);
      toast.success("Note mise à jour");
      setIsEditing(false);
    },
    onError: (error: Error) => {
      setValue(originalGrade); // Revert to original on error
      console.error("Grade update error:", error);
      toast.error(`Erreur: ${error.message || "Erreur lors de la mise à jour"}`);
    },
  });

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isEditing]);

  useEffect(() => {
    setOriginalGrade(grade);
  }, [grade]);

  const handleSave = () => {
    if (value !== grade) {
      updateMutation.mutate(value);
    } else {
      setIsEditing(false);
    }
  };

  const handleReset = () => {
    setValue(originalGrade);
    updateMutation.mutate(originalGrade);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setValue(originalGrade);
      setIsEditing(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {hasDisagreement && <AlertTriangle className="h-4 w-4 text-warning" />}
      {isEditing ? (
        <div className="flex items-center gap-1">
          <input
            ref={inputRef}
            type="number"
            min={0}
            max={maxPoints}
            step={0.5}
            value={value}
            onChange={(e) => setValue(parseFloat(e.target.value) || 0)}
            onKeyDown={handleKeyDown}
            onBlur={handleSave}
            className="w-16 px-2 py-1 text-sm border rounded"
          />
          <span className="text-sm text-muted-foreground">/ {maxPoints}</span>
        </div>
      ) : (
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsEditing(true)}
            className="text-sm font-medium hover:bg-muted px-2 py-1 rounded transition-colors"
            title="Cliquez sur une note pour la modifier"
          >
            {grade} / {maxPoints}
          </button>
          {grade !== originalGrade && (
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6"
              onClick={handleReset}
              title="Restaurer la note originale de l'IA"
            >
              <RotateCcw className="h-3 w-3 text-muted-foreground" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
