"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { SessionStatus } from "@/components/grading/session-status";
import { formatDate } from "@/lib/utils";
import { Eye, Trash2, Calendar, FileText, Users } from "lucide-react";
import type { Session } from "@/lib/types";

interface SessionCardProps {
  session: Session;
  onDelete: (sessionId: string) => void;
}

export function SessionCard({ session, onDelete }: SessionCardProps) {
  // Determine the primary action based on session status
  const isDiagnostic = session.status === "diagnostic";
  const isCorrection = session.status === "correction";
  const isInProgress = isDiagnostic || isCorrection;
  const actionText = isDiagnostic
    ? "Continuer le diagnostic"
    : isCorrection
    ? "Voir la correction"
    : "Voir résultats";
  const actionIcon = isInProgress ? (
    <FileText className="h-3.5 w-3.5 mr-1" />
  ) : (
    <Eye className="h-3.5 w-3.5 mr-1" />
  );

  // For diagnostic status, link to resume; for others, go to session page
  const actionHref = isDiagnostic
    ? `/sessions/new?resume=${session.session_id}`
    : `/sessions/${session.session_id}`;

  return (
    <Card className="hover:shadow-md transition-all duration-200 rounded-lg">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <Link
              href={`/sessions/${session.session_id}`}
              className="font-medium hover:text-purple-600 transition-colors block truncate"
            >
              {session.subject || session.session_id}
            </Link>
            {session.topic && (
              <p className="text-sm text-muted-foreground truncate">{session.topic}</p>
            )}
          </div>
          <SessionStatus status={session.status} />
        </div>

        <div className="grid grid-cols-3 gap-2 mb-3 text-sm">
          <div className="flex items-center gap-1 text-muted-foreground">
            <FileText className="h-3.5 w-3.5" />
            <span>{session.graded_count}/{session.copies_count}</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Users className="h-3.5 w-3.5" />
            <span>{session.average_score?.toFixed(1) || "-"}/{session.max_score?.toFixed(0) || "-"}</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Calendar className="h-3.5 w-3.5" />
            <span className="text-xs">{formatDate(session.created_at)}</span>
          </div>
        </div>

        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            className="flex-1 hover:bg-purple-50 hover:border-purple-200 hover:text-purple-700"
            asChild
          >
            <Link href={actionHref}>
              {actionIcon}
              {actionText}
            </Link>
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
            onClick={() => onDelete(session.session_id)}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

interface SessionCardsGridProps {
  sessions: Session[];
  onDelete: (sessionId: string) => void;
}

export function SessionCardsGrid({ sessions, onDelete }: SessionCardsGridProps) {
  if (sessions.length === 0) {
    return (
      <div className="text-center py-12">
        <FileText className="h-16 w-16 mx-auto text-muted-foreground/50 mb-4" />
        <h3 className="font-semibold text-lg mb-2">Aucune correction</h3>
        <p className="text-muted-foreground mb-4">
          Créez votre première correction pour commencer
        </p>
        <Button asChild className="bg-purple-600 hover:bg-purple-700">
          <Link href="/sessions/new">Nouvelle correction</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
      {sessions.map((session) => (
        <SessionCard key={session.session_id} session={session} onDelete={onDelete} />
      ))}
    </div>
  );
}
