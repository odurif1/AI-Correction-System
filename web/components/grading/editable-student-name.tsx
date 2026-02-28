"use client";

import { useState, useRef, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { toast } from "sonner";

interface EditableStudentNameProps {
  sessionId: string;
  copyId: string;
  studentName?: string;
}

export function EditableStudentName({
  sessionId,
  copyId,
  studentName,
}: EditableStudentNameProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(studentName || "");
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationFn: (newName: string) =>
      api.updateStudentName(sessionId, copyId, newName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      toast.success("Nom mis à jour");
      setIsEditing(false);
    },
    onError: (error: Error) => {
      setValue(studentName || "");
      console.error("Student name update error:", error);
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
    setValue(studentName || "");
  }, [studentName]);

  const handleSave = () => {
    const trimmedValue = value.trim();
    if (trimmedValue && trimmedValue !== studentName) {
      updateMutation.mutate(trimmedValue);
    } else {
      setIsEditing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setValue(studentName || "");
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
        className="px-2 py-1 text-sm border rounded w-full min-w-[100px]"
        placeholder="Nom de l'élève"
      />
    );
  }

  return (
    <button
      onClick={() => setIsEditing(true)}
      className="text-sm font-medium hover:bg-muted px-2 py-1 rounded transition-colors text-left w-full"
      title="Cliquez pour modifier le nom"
    >
      {studentName || copyId}
    </button>
  );
}
