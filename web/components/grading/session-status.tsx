"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Loader2,
  CheckCircle2,
  AlertCircle,
  Search,
} from "lucide-react";

interface SessionStatusProps {
  status: string;
  className?: string;
}

const statusConfig: Record<
  string,
  { label: string; icon: React.ReactNode; variant: "default" | "secondary" | "destructive" | "outline" | "success" | "warning" }
> = {
  diagnostic: {
    label: "Diagnostic",
    icon: <Search className="h-3 w-3 mr-1" />,
    variant: "secondary",
  },
  correction: {
    label: "Correction en cours",
    icon: <Loader2 className="h-3 w-3 mr-1 animate-spin" />,
    variant: "default",
  },
  complete: {
    label: "Terminée",
    icon: <CheckCircle2 className="h-3 w-3 mr-1" />,
    variant: "success",
  },
  error: {
    label: "Erreur",
    icon: <AlertCircle className="h-3 w-3 mr-1" />,
    variant: "destructive",
  },
};

export function SessionStatus({ status, className }: SessionStatusProps) {
  const config = statusConfig[status.toLowerCase()] || {
    label: status,
    icon: null,
    variant: "outline" as const,
  };

  return (
    <Badge variant={config.variant} className={cn(className)}>
      {config.icon}
      {config.label}
    </Badge>
  );
}
