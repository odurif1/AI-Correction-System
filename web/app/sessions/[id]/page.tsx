"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { SessionStatus } from "@/components/grading/session-status";
import { ProgressGrid } from "@/components/grading/progress-grid";
import { ScoreDistribution } from "@/components/grading/score-distribution";
import { DisagreementCard } from "@/components/grading/disagreement-card";
import { ExportButton } from "@/components/export-button";
import { EditableGradeCell } from "@/components/grading/editable-grade-cell";
import { useProgressSocket } from "@/lib/websocket";
import { useRotatingMessage } from "@/lib/waiting-messages";
import { api } from "@/lib/api";
import { formatDate } from "@/lib/utils";
import {
  ArrowLeft,
  Play,
  Loader2,
  Users,
  BarChart3,
  AlertTriangle,
  X,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import type { Disagreement, SessionDetail, Analytics } from "@/lib/types";

export default function SessionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.id as string;

  const [disagreements, setDisagreements] = useState<Disagreement[]>([]);
  const [isGrading, setIsGrading] = useState(false);
  const [sortColumn, setSortColumn] = useState<string>("copy_id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // Fetch session data
  const { data: session, isLoading } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch analytics
  const { data: analytics } = useQuery({
    queryKey: ["analytics", sessionId],
    queryFn: () => api.getAnalytics(sessionId),
    enabled: session?.status === "complete",
  });

  // Fetch disagreements
  const { data: disagreementData } = useQuery({
    queryKey: ["disagreements", sessionId],
    queryFn: () => api.getDisagreements(sessionId),
    enabled: session?.status === "complete",
  });

  // WebSocket for real-time progress
  const progress = useProgressSocket({
    sessionId,
    onComplete: () => {
      // Refetch data when grading completes
      router.refresh();
    },
  });

  // Calculate agreement rate for dual-LLM grading
  const agreementRate = useMemo(() => {
    const graded = progress.copies.filter(c => c.status === 'done');
    if (graded.length === 0) return null;
    const agreed = graded.filter(c => c.agreement === true).length;
    return Math.round((agreed / graded.length) * 100);
  }, [progress.copies]);

  // Update grading state based on session
  useEffect(() => {
    if (session) {
      const gradingInProgress = session.status === "grading" || session.status === "analyzing";
      setIsGrading(gradingInProgress);
    }
  }, [session]);

  // Start grading mutation
  const startGradingMutation = useMutation({
    mutationFn: () => api.startGrading(sessionId),
    onSuccess: () => {
      setIsGrading(true);
    },
  });

  // Cancel grading mutation (delete session)
  const cancelMutation = useMutation({
    mutationFn: () => api.deleteSession(sessionId),
    onSuccess: () => {
      router.push("/dashboard");
    },
  });

  // Resolve disagreement mutation
  const resolveMutation = useMutation({
    mutationFn: ({
      questionId,
      decision,
    }: {
      questionId: string;
      decision: any;
    }) => api.resolveDisagreement(sessionId, questionId, decision),
    onSuccess: (_, { questionId }) => {
      // Remove resolved disagreement from list
      setDisagreements((prev) =>
        prev.filter((d) => d.question_id !== questionId)
      );
    },
  });

  useEffect(() => {
    if (disagreementData) {
      setDisagreements(disagreementData);
    }
  }, [disagreementData]);

  // Auto-navigation on completion
  useEffect(() => {
    if (session?.status === "complete" && isGrading) {
      // Delay slightly to show final status
      const timeout = setTimeout(() => {
        router.push(`/sessions/${sessionId}?tab=review`);
        setIsGrading(false);
      }, 1500);
      return () => clearTimeout(timeout);
    }
  }, [session?.status, isGrading, sessionId, router]);

  // Handle cancel button
  const handleCancel = async () => {
    if (confirm("Êtes-vous sûr ? Cela annulera toute la progression.")) {
      await cancelMutation.mutateAsync();
    }
  };

  // Get rotating waiting message
  const waitingMessage = useRotatingMessage(5000);

  // Sorting helper
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(column);
      setSortDirection("asc");
    }
  };

  const getSortedCopies = () => {
    if (!session?.graded_copies) return [];

    return [...session.graded_copies].sort((a, b) => {
      let aValue: any;
      let bValue: any;

      switch (sortColumn) {
        case "copy_id":
          aValue = a.copy_id;
          bValue = b.copy_id;
          break;
        case "total_score":
          aValue = a.total_score;
          bValue = b.total_score;
          break;
        case "percentage":
          aValue = (a.total_score / a.max_score) * 100;
          bValue = (b.total_score / b.max_score) * 100;
          break;
        default:
          // For question columns, access grades object
          aValue = a.grades[sortColumn] || 0;
          bValue = b.grades[sortColumn] || 0;
      }

      if (typeof aValue === "string") {
        return sortDirection === "asc"
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
    });
  };

  const sortedCopies = getSortedCopies();

  // Get question IDs from the first graded copy or question_weights
  const questionIds = session?.question_weights
    ? Object.keys(session.question_weights)
    : (session?.graded_copies[0]?.grades ? Object.keys(session.graded_copies[0].grades) : []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </main>
        <Footer />
      </div>
    );
  }

  if (!session) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container py-8">
          <div className="text-center">
            <h1 className="text-2xl font-bold mb-4">Session not found</h1>
            <Button asChild>
              <Link href="/dashboard">Back to Dashboard</Link>
            </Button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  // Use state variable for isGrading to track during grading
  // Derived from session status but also managed during transitions
  const isGradingActive = session.status === "grading" || session.status === "analyzing";
  const isComplete = session.status === "complete";
  const hasDisagreements = disagreements.filter((d) => !d.resolved).length > 0;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8">
        <Button variant="ghost" className="mb-6" asChild>
          <Link href="/dashboard">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Link>
        </Button>

        {/* Session Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-bold tracking-tight">
                Session {sessionId}
              </h1>
              <SessionStatus status={session.status} />
              {progress.connected && (
                <Badge variant="outline" className="text-success border-success">
                  Live
                </Badge>
              )}
            </div>
            <div className="text-muted-foreground">
              {session.subject && <span>{session.subject}</span>}
              {session.topic && <span> - {session.topic}</span>}
              <span className="mx-2">|</span>
              Created {formatDate(session.created_at)}
            </div>
          </div>

          <div className="flex gap-2">
            {!isGradingActive && !isComplete && (
              <Button
                onClick={() => startGradingMutation.mutate()}
                disabled={startGradingMutation.isPending}
                className="bg-purple-600 hover:bg-purple-700"
              >
                {startGradingMutation.isPending ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Play className="h-4 w-4 mr-2" />
                )}
                Start Grading
              </Button>
            )}
            {isComplete && (
              <ExportButton sessionId={sessionId} />
            )}
          </div>
        </div>

        {/* Progress Grid (during grading) */}
        {isGradingActive && (
          <Card className="mb-8 border-purple-200 dark:border-purple-900">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
                    Grading in Progress
                  </CardTitle>
                  <CardDescription className="mt-1">
                    {waitingMessage}
                  </CardDescription>
                </div>
                {/* Cancel button */}
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleCancel}
                  disabled={cancelMutation.isPending}
                  className="gap-1"
                  aria-label="Annuler la correction"
                >
                  {cancelMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <X className="h-4 w-4" />
                  )}
                  Cancel
                </Button>
              </div>
              {/* Progress summary */}
              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2" role="status" aria-live="polite">
                <span>
                  {progress.completedCopies} of {progress.totalCopies || session.copies_count} copies
                  completed
                </span>
                {agreementRate !== null && (
                  <span className="flex items-center gap-1">
                    <Badge variant="outline" className="text-xs">
                      {agreementRate}% accord
                    </Badge>
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ProgressGrid
                copies={progress.copies}
                totalCopies={progress.totalCopies || session.copies_count}
              />
            </CardContent>
          </Card>
        )}

        {/* Stats Cards */}
        <div className="grid gap-4 md:grid-cols-4 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Total Copies
              </CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{session.copies_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Graded</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{session.graded_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Average Score
              </CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {session.average_score?.toFixed(1) || "-"}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Disagreements
              </CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {disagreements.filter((d) => !d.resolved).length}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tabs for Review/Analytics/Disagreements */}
        {isComplete && (
          <Tabs defaultValue="review" className="space-y-6">
            <TabsList>
              <TabsTrigger value="review">Review</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              {hasDisagreements && (
                <TabsTrigger value="disagreements">
                  Disagreements
                  <Badge variant="warning" className="ml-2">
                    {disagreements.filter((d) => !d.resolved).length}
                  </Badge>
                </TabsTrigger>
              )}
            </TabsList>

            {/* Review Tab */}
            <TabsContent value="review">
              <Card>
                <CardHeader>
                  <CardTitle>Review Grades</CardTitle>
                  <CardDescription>
                    Click any grade to edit. Changes auto-save.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="rounded-md border overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead
                            className="cursor-pointer hover:bg-muted sticky left-0 bg-background z-10 min-w-[120px]"
                            onClick={() => handleSort("copy_id")}
                          >
                            <div className="flex items-center gap-1">
                              Copy / Student
                              {sortColumn === "copy_id" && (
                                sortDirection === "asc" ? (
                                  <ChevronUp className="h-4 w-4 text-purple-600" />
                                ) : (
                                  <ChevronDown className="h-4 w-4 text-purple-600" />
                                )
                              )}
                            </div>
                          </TableHead>
                          {questionIds.map((questionId) => (
                            <TableHead
                              key={questionId}
                              className="cursor-pointer hover:bg-muted text-center min-w-[100px]"
                              onClick={() => handleSort(questionId)}
                            >
                              <div className="flex items-center justify-center gap-1">
                                {questionId}
                                {sortColumn === questionId && (
                                  sortDirection === "asc" ? (
                                    <ChevronUp className="h-4 w-4 text-purple-600" />
                                  ) : (
                                    <ChevronDown className="h-4 w-4 text-purple-600" />
                                  )
                                )}
                              </div>
                            </TableHead>
                          ))}
                          <TableHead
                            className="cursor-pointer hover:bg-muted text-center min-w-[80px]"
                            onClick={() => handleSort("total_score")}
                          >
                            <div className="flex items-center justify-center gap-1">
                              Total
                              {sortColumn === "total_score" && (
                                sortDirection === "asc" ? (
                                  <ChevronUp className="h-4 w-4 text-purple-600" />
                                ) : (
                                  <ChevronDown className="h-4 w-4 text-purple-600" />
                                )
                              )}
                            </div>
                          </TableHead>
                          <TableHead
                            className="cursor-pointer hover:bg-muted text-center min-w-[60px]"
                            onClick={() => handleSort("percentage")}
                          >
                            <div className="flex items-center justify-center gap-1">
                              %
                              {sortColumn === "percentage" && (
                                sortDirection === "asc" ? (
                                  <ChevronUp className="h-4 w-4 text-purple-600" />
                                ) : (
                                  <ChevronDown className="h-4 w-4 text-purple-600" />
                                )
                              )}
                            </div>
                          </TableHead>
                          <TableHead className="text-center min-w-[100px]">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedCopies.map((graded) => {
                          // Get grading audit to check for disagreements
                          const gradingAudit = graded.grading_audit as any;

                          return (
                            <TableRow key={graded.copy_id} className="hover:bg-muted/50">
                              <TableCell className="font-medium sticky left-0 bg-background">
                                {graded.copy_id}
                              </TableCell>
                              {questionIds.map((questionId) => {
                                // Check if this specific question has a disagreement
                                const questionHasDisagreement = gradingAudit?.questions?.[questionId]?.resolution?.agreement === false;
                                const maxPoints = session.question_weights?.[questionId] || 0;

                                return (
                                  <TableCell key={questionId} className="text-center">
                                    <EditableGradeCell
                                      sessionId={sessionId}
                                      copyId={graded.copy_id}
                                      questionId={questionId}
                                      grade={graded.grades[questionId] || 0}
                                      maxPoints={maxPoints}
                                      hasDisagreement={questionHasDisagreement}
                                    />
                                  </TableCell>
                                );
                              })}
                              <TableCell className="text-center font-medium">
                                {graded.total_score.toFixed(1)}/{graded.max_score}
                              </TableCell>
                              <TableCell className="text-center">
                                {((graded.total_score / graded.max_score) * 100).toFixed(1)}%
                              </TableCell>
                              <TableCell className="text-center">
                                <Button variant="ghost" size="sm" asChild>
                                  <a
                                    href={`/api/sessions/${sessionId}/copies/${graded.copy_id}/pdf`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="flex items-center gap-1"
                                  >
                                    <ExternalLink className="h-3 w-3" />
                                    View PDF
                                  </a>
                                </Button>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Analytics Tab */}
            <TabsContent value="analytics">
              {analytics && (
                <div className="grid gap-6 md:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle>Statistics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <dl className="grid grid-cols-2 gap-4">
                        <div>
                          <dt className="text-sm text-muted-foreground">Mean</dt>
                          <dd className="text-2xl font-bold">
                            {analytics.mean_score.toFixed(2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm text-muted-foreground">Median</dt>
                          <dd className="text-2xl font-bold">
                            {analytics.median_score.toFixed(2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm text-muted-foreground">Min</dt>
                          <dd className="text-2xl font-bold">
                            {analytics.min_score.toFixed(2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm text-muted-foreground">Max</dt>
                          <dd className="text-2xl font-bold">
                            {analytics.max_score.toFixed(2)}
                          </dd>
                        </div>
                        <div className="col-span-2">
                          <dt className="text-sm text-muted-foreground">
                            Standard Deviation
                          </dt>
                          <dd className="text-2xl font-bold">
                            {analytics.std_dev.toFixed(2)}
                          </dd>
                        </div>
                      </dl>
                    </CardContent>
                  </Card>

                  <ScoreDistribution
                    distribution={analytics.score_distribution}
                    title="Score Distribution"
                  />
                </div>
              )}
            </TabsContent>

            {/* Disagreements Tab */}
            {hasDisagreements && (
              <TabsContent value="disagreements">
                <div className="space-y-6">
                  {disagreements
                    .filter((d) => !d.resolved)
                    .map((disagreement) => (
                      <DisagreementCard
                        key={`${disagreement.copy_id}-${disagreement.question_id}`}
                        disagreement={disagreement}
                        onResolve={async (decision) => {
                          await resolveMutation.mutateAsync({
                            questionId: disagreement.question_id,
                            decision,
                          });
                        }}
                      />
                    ))}
                </div>
              </TabsContent>
            )}
          </Tabs>
        )}
      </main>

      <Footer />
    </div>
  );
}
