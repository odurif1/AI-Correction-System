"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "sonner";
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
      toast.success("Nom de l'examen mis à jour");
      setIsEditing(false);
    },
    onError: (error: Error) => {
      setValue(subject || "");
      console.error("Session subject update error:", error);
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
        className="px-3 py-1 text-3xl font-bold tracking-tight border rounded-lg bg-background w-full max-w-md"
        placeholder="Nom de l'examen"
      />
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="group flex items-center gap-2 text-3xl font-bold tracking-tight hover:bg-muted/50 px-3 py-1 rounded-lg transition-colors text-left"
      title="Cliquez pour modifier le nom de l'examen"
    >
      {subject || fallback || "Examen sans nom"}
      <Pencil className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
    </button>
  );
}
