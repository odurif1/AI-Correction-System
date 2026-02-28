"use client";

import { useState } from "react";
import { Pencil } from "lucide-react";
import { api } from "@/lib/api";

interface EditableQuestionWeightProps {
  sessionId: string;
  questionId: string;
  weight: number;
  onWeightChange?: (questionId: string, newWeight: number) => void;
}

export function EditableQuestionWeight({
  sessionId,
  questionId,
  weight,
  onWeightChange,
}: EditableQuestionWeightProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(weight.toString());
  const [isLoading, setIsLoading] = useState(false);

  const handleSave = async () => {
    const newWeight = parseFloat(value);
    if (isNaN(newWeight) || newWeight < 0) {
      setValue(weight.toString());
      setIsEditing(false);
      return;
    }

    if (newWeight === weight) {
      setIsEditing(false);
      return;
    }

    setIsLoading(true);
    try {
      const result = await api.updateQuestionWeight(sessionId, questionId, newWeight);
      onWeightChange?.(questionId, newWeight);
    } catch (error) {
      setValue(weight.toString());
    } finally {
      setIsLoading(false);
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSave();
    } else if (e.key === "Escape") {
      setValue(weight.toString());
      setIsEditing(false);
    }
  };

  if (isEditing) {
    return (
      <input
        type="number"
        min={0}
        step={0.5}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={handleSave}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        className="w-12 text-center text-sm border rounded px-1 py-0.5 focus:outline-none focus:ring-2 focus:ring-purple-500"
        autoFocus
      />
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="group flex items-center gap-0.5 hover:text-purple-600 transition-colors"
      title="Cliquer pour modifier le barème"
    >
      <span>({weight} pts)</span>
      <Pencil className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  );
}
