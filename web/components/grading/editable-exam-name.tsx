"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Pencil } from "lucide-react";

interface EditableExamNameProps {
  sessionId: string;
  subject?: string;
  fallback?: string;
}

export function EditableExamName({
  sessionId,
  subject,
  fallback,
}: EditableExamNameProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(subject || "");
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationFn: (newSubject: string) =>
      api.updateSessionSubject(sessionId, newSubject),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      setIsEditing(false);
    },
    onError: (error: Error) => {
      setValue(subject || "");
      console.error("Session subject update error:", error);
    },
  });

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isEditing]);

  useEffect(() => {
    setValue(subject || "");
  }, [subject]);

  const handleSave = () => {
    const trimmedValue = value.trim();
    if (trimmedValue && trimmedValue !== subject) {
      updateMutation.mutate(trimmedValue);
    } else {
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setValue(subject || "");
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
        onKeyDown={handleKeyDown}
        onBlur={handleSave}
        className="px-2 py-0.5 text-base font-semibold border border-purple-300 rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-purple-200"
        placeholder="Nom de l'examen"
        style={{ width: `${Math.max(value.length + 2, 10)}ch` }}
      />
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="group inline-flex items-center gap-1.5 text-base font-semibold hover:text-purple-600 transition-colors"
      title="Cliquez pour modifier le nom de l'examen"
    >
      {subject || fallback || "Examen sans nom"}
      <Pencil className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  );
}
