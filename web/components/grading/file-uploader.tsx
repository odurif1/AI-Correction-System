"use client";

import { useCallback, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { Upload, FileText, X, AlertCircle } from "lucide-react";

export interface FileWithProgress {
  file: File;
  progress: number;
  error?: string;
}

interface FileUploaderProps {
  onFilesSelected: (files: File[]) => void;
  accept?: string;
  multiple?: boolean;
  maxFiles?: number;
  className?: string;
  // New props for multi-file upload with progress
  files?: FileWithProgress[];
  uploading?: boolean;
}

export function FileUploader({
  onFilesSelected,
  accept = ".pdf",
  multiple = true,
  maxFiles = 50,
  className,
  files: externalFiles,
  uploading = false,
}: FileUploaderProps) {
  const [internalFiles, setInternalFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  // Use external files if provided (for progress tracking), otherwise use internal state
  const displayFiles = externalFiles || internalFiles.map(file => ({ file, progress: 0, error: undefined as string | undefined }));

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const droppedFiles = Array.from(e.dataTransfer.files).filter((file) =>
        file.name.toLowerCase().endsWith(".pdf")
      );

      const newFiles = multiple
        ? [...internalFiles, ...droppedFiles].slice(0, maxFiles)
        : droppedFiles.slice(0, 1);

      setInternalFiles(newFiles);
      onFilesSelected(newFiles);
    },
    [internalFiles, maxFiles, multiple, onFilesSelected]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const selectedFiles = Array.from(e.target.files);
        const newFiles = multiple
          ? [...internalFiles, ...selectedFiles].slice(0, maxFiles)
          : selectedFiles.slice(0, 1);

        setInternalFiles(newFiles);
        onFilesSelected(newFiles);
      }
    },
    [internalFiles, maxFiles, multiple, onFilesSelected]
  );

  const removeFile = (index: number) => {
    const newFiles = internalFiles.filter((_, i) => i !== index);
    setInternalFiles(newFiles);
    onFilesSelected(newFiles);
  };

  return (
    <div className={cn("space-y-4", className)}>
      <Card
        className={cn(
          "border-dashed cursor-pointer transition-all hover:shadow-md hover:border-purple-300",
          isDragging && "border-primary bg-primary/5 shadow-md"
        )}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <CardContent className="flex flex-col items-center justify-center py-10">
          <Upload className="h-10 w-10 text-muted-foreground mb-4" />
          <p className="mb-2 text-sm text-muted-foreground">
            <span className="font-semibold">Click to upload</span> or drag and
            drop
          </p>
          <p className="text-xs text-muted-foreground">
            PDF files {multiple && `(max ${maxFiles} files)`}
          </p>
          <input
            type="file"
            accept={accept}
            multiple={multiple}
            onChange={handleFileInput}
            className="hidden"
            id="file-upload"
          />
          <Button variant="outline" className="mt-4 min-h-[44px] min-w-[120px] transition-colors hover:bg-purple-50 hover:border-purple-300" asChild>
            <label htmlFor="file-upload" className="cursor-pointer">
              Select Files
            </label>
          </Button>
        </CardContent>
      </Card>

      {displayFiles.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">
            {displayFiles.length} file{displayFiles.length !== 1 ? "s" : ""} selected
          </p>
          <div className="grid gap-2">
            {displayFiles.map((fileWithProgress, index) => {
              const file = fileWithProgress.file;
              const progress = fileWithProgress.progress || 0;
              const error = fileWithProgress.error;

              return (
                <div
                  key={`${file.name}-${index}`}
                  className={cn(
                    "p-2 bg-muted rounded-md border transition-colors",
                    error && "border-destructive/50 bg-destructive/5"
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      <span className="text-sm truncate">
                        {file.name}
                      </span>
                      <span className="text-xs text-muted-foreground flex-shrink-0">
                        ({(file.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 flex-shrink-0 min-h-[44px] min-w-[44px] transition-colors hover:bg-destructive/10 hover:text-destructive"
                      onClick={() => removeFile(index)}
                      disabled={uploading}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  {/* Progress bar */}
                  {uploading && progress > 0 && (
                    <div className="space-y-1">
                      <Progress value={progress} className="h-1" />
                      <p className="text-xs text-muted-foreground">
                        {progress}%
                      </p>
                    </div>
                  )}

                  {/* Error message */}
                  {error && (
                    <div className="flex items-center gap-1 text-destructive text-xs mt-1">
                      <AlertCircle className="h-3 w-3" />
                      <span>{error}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
