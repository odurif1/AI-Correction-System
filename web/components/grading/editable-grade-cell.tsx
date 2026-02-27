"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { Check, X, AlertTriangle } from "lucide-react";
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
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationFn: (newGrade: number) =>
      api.updateGrade(sessionId, copyId, questionId, { new_grade: newGrade }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      toast.success("Note mise à jour");
      setIsEditing(false);
    },
    onError: () => {
      setValue(grade); // Revert on error
      toast.error("Erreur lors de la mise à jour");
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

  const handleCancel = () => {
    setValue(grade);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") handleCancel();
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
            className="w-16 px-2 py-1 text-sm border rounded"
          />
          <span className="text-sm text-muted-foreground">/ {maxPoints}</span>
          <Button size="icon" variant="ghost" className="h-6 w-6" onClick={handleSave}>
            <Check className="h-3 w-3 text-success" />
          </Button>
          <Button size="icon" variant="ghost" className="h-6 w-6" onClick={handleCancel}>
            <X className="h-3 w-3 text-destructive" />
          </Button>
        </div>
      ) : (
        <button
          onClick={() => setIsEditing(true)}
          className="text-sm font-medium hover:bg-muted px-2 py-1 rounded transition-colors"
        >
          {grade} / {maxPoints}
        </button>
      )}
    </div>
  );
}
