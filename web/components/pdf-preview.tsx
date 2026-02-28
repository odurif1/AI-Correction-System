"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileText, ZoomIn, ZoomOut, Download, X } from "lucide-react";

interface PdfPreviewProps {
  file: File | null;
  url?: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PdfPreview({ file, url, open, onOpenChange }: PdfPreviewProps) {
  const [scale, setScale] = useState(1);

  const pdfUrl = file ? URL.createObjectURL(file) : url;

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.25, 3));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.25, 0.5));

  if (!pdfUrl) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col">
        <DialogHeader className="flex-row items-center justify-between">
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            {file?.name || "PDF Preview"}
          </DialogTitle>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <span className="text-sm w-12 text-center">{Math.round(scale * 100)}%</span>
            <Button variant="outline" size="icon" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            {file && (
              <Button variant="outline" size="icon" asChild>
                <a href={pdfUrl} download={file.name}>
                  <Download className="h-4 w-4" />
                </a>
              </Button>
            )}
          </div>
        </DialogHeader>
        <ScrollArea className="flex-1 bg-muted/50 rounded-lg">
          <div className="flex justify-center p-4">
            <iframe
              src={pdfUrl}
              className="border-0 rounded shadow-lg"
              style={{
                width: `${scale * 100}%`,
                height: `${scale * 70}vh`,
                minHeight: "400px",
              }}
              title="PDF Preview"
            />
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}

// Compact file list with preview
interface FileWithPreview {
  file: File;
  id: string;
}

interface FileListWithPreviewProps {
  files: File[];
  onRemove?: (index: number) => void;
}

export function FileListWithPreview({ files, onRemove }: FileListWithPreviewProps) {
  const [previewFile, setPreviewFile] = useState<File | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  const handlePreview = (file: File) => {
    setPreviewFile(file);
    setPreviewOpen(true);
  };

  if (files.length === 0) return null;

  return (
    <>
      <div className="space-y-2">
        <p className="text-sm font-medium">{files.length} file(s) selected</p>
        <div className="grid gap-2 max-h-60 overflow-y-auto">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center justify-between p-2 bg-muted rounded-md hover:bg-muted/80 transition-colors"
            >
              <button
                className="flex items-center gap-2 flex-1 text-left hover:text-primary transition-colors"
                onClick={() => handlePreview(file)}
              >
                <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <span className="text-sm truncate max-w-[200px]">{file.name}</span>
                <span className="text-xs text-muted-foreground">
                  ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </button>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => handlePreview(file)}
                >
                  Preview
                </Button>
                {onRemove && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-destructive hover:text-destructive"
                    onClick={() => onRemove(index)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <PdfPreview
        file={previewFile}
        open={previewOpen}
        onOpenChange={setPreviewOpen}
      />
    </>
  );
}
