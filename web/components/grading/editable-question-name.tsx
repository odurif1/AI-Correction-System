"use client";

import { useState, useRef, useEffect } from "react";
import { Pencil } from "lucide-react";
import { api } from "@/lib/api";

interface EditableQuestionNameProps {
  sessionId: string;
  questionId: string;
  displayName: string;
  onNameChange?: () => void;
}

export function EditableQuestionName({
  sessionId,
  questionId,
  displayName,
  onNameChange,
}: EditableQuestionNameProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(displayName);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isEditing]);

  const handleSave = async () => {
    const newName = value.trim();
    if (!newName) {
      setValue(displayName);
      setIsEditing(false);
      return;
    }

    if (newName === displayName) {
      setIsEditing(false);
      return;
    }

    setIsLoading(true);
    try {
      await api.updateQuestionName(sessionId, questionId, newName);
      onNameChange?.();
    } catch (error) {
      setValue(displayName);
    } finally {
      setIsLoading(false);
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSave();
    } else if (e.key === "Escape") {
      setValue(displayName);
      setIsEditing(false);
    }
  };

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={handleSave}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        className="w-20 text-center text-xs font-semibold border border-purple-300 rounded px-1 py-0.5 focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white"
        maxLength={50}
      />
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="group flex items-center gap-0.5 hover:text-purple-600 transition-colors"
      title="Cliquer pour renommer la question"
    >
      <span>{displayName}</span>
      <Pencil className="h-2.5 w-2.5 opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  );
}
