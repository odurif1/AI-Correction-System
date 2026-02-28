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
  /** The original LLM grade - used for reset functionality */
  originalLLMGrade?: number;
}

export function EditableGradeCell({
  sessionId,
  copyId,
  questionId,
  grade,
  maxPoints,
  hasDisagreement,
  originalLLMGrade,
}: EditableGradeCellProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(grade);
  const [disagreementAcknowledged, setDisagreementAcknowledged] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // Track the initial grade from when component first mounted (the LLM's original grade)
  // This ref never changes - it's the "reset to" value
  const initialGradeRef = useRef<number | null>(null);

  // Set initial grade only once when component mounts
  useEffect(() => {
    if (initialGradeRef.current === null) {
      initialGradeRef.current = originalLLMGrade ?? grade;
    }
  }, [originalLLMGrade, grade]);

  const updateMutation = useMutation({
    mutationFn: (newGrade: number) =>
      api.updateGrade(sessionId, copyId, questionId, { new_grade: newGrade }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      toast.success("Note mise à jour");
      setIsEditing(false);
    },
    onError: (error: Error) => {
      setValue(grade); // Revert to current saved grade on error
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

  const handleSave = () => {
    if (value !== grade) {
      updateMutation.mutate(value);
    } else {
      setIsEditing(false);
    }
  };

  const handleReset = () => {
    const resetValue = initialGradeRef.current ?? grade;
    setValue(resetValue);
    updateMutation.mutate(resetValue);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setValue(grade);
      setIsEditing(false);
    }
  };

  // Show reset button only if current grade differs from initial LLM grade
  const showResetButton = initialGradeRef.current !== null && grade !== initialGradeRef.current;

  // Show warning only if there's a disagreement and user hasn't acknowledged it
  const showWarning = hasDisagreement && !disagreementAcknowledged;

  const handleClick = () => {
    setDisagreementAcknowledged(true);
    setIsEditing(true);
  };

  return (
    <div className="flex items-center gap-2">
      {showWarning && <AlertTriangle className="h-4 w-4 text-warning" />}
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
            onClick={handleClick}
            className="text-sm font-medium hover:bg-muted px-2 py-1 rounded transition-colors"
            title="Cliquez sur une note pour la modifier"
          >
            {grade} / {maxPoints}
          </button>
          {showResetButton && (
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
