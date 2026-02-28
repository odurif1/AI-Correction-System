"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { RotateCcw, AlertTriangle, Check } from "lucide-react";
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
  const [showSuccess, setShowSuccess] = useState(false);
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
      setIsEditing(false);
      // Show success animation
      setShowSuccess(true);
      setTimeout(() => setShowSuccess(false), 1500);
    },
    onError: (error: Error) => {
      setValue(grade); // Revert to current saved grade on error
      console.error("Grade update error:", error);
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
            className="w-14 px-2 py-1.5 text-sm font-medium text-center border border-purple-200 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white"
          />
        </div>
      ) : (
        <div className="flex items-center gap-1 relative">
          <button
            onClick={handleClick}
            className={`inline-flex items-center gap-0.5 px-2.5 py-1.5 rounded-md text-sm font-medium transition-all duration-200 group/btn ${
              showSuccess
                ? 'bg-green-100 text-green-700 ring-2 ring-green-300 ring-offset-1'
                : 'bg-slate-100/80 hover:bg-purple-100 hover:text-purple-700 text-slate-700'
            }`}
            title="Cliquez sur une note pour la modifier"
          >
            {showSuccess ? (
              <>
                <Check className="h-4 w-4 animate-in zoom-in duration-200" />
                <span>{grade}</span>
              </>
            ) : (
              <>
                <span>{grade}</span>
                <span className="text-slate-400 group-hover/btn:text-purple-400">/</span>
                <span className="text-slate-500 text-xs">{maxPoints}</span>
              </>
            )}
          </button>
          {showResetButton && (
            <Button
              size="icon"
              variant="ghost"
              className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-amber-100 hover:text-amber-600"
              onClick={handleReset}
              title="Restaurer la note originale de l'IA"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
