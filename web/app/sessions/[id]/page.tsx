"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
import { EditableStudentName } from "@/components/grading/editable-student-name";
import { EditableExamName } from "@/components/grading/editable-exam-name";
import { EditableQuestionWeight } from "@/components/grading/editable-question-weight";
import { EditableQuestionName } from "@/components/grading/editable-question-name";
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
  Search,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import type { Disagreement, SessionDetail, Analytics } from "@/lib/types";

export default function SessionDetailPage() {
  const params = useParams();
  const router = useRouter();
  const queryClient = useQueryClient();
  const sessionId = params.id as string;

  const [disagreements, setDisagreements] = useState<Disagreement[]>([]);
  const [isGrading, setIsGrading] = useState(false);
  const [sortColumn, setSortColumn] = useState<string>("copy_id");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<string>("review");

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
      const gradingInProgress = session.status === "correction";
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
      copyId,
      questionId,
      decision,
    }: {
      copyId: string;
      questionId: string;
      decision: any;
    }) => api.resolveDisagreement(sessionId, copyId, questionId, decision),
    onSuccess: (_, { questionId }) => {
      // Remove resolved disagreement from list
      setDisagreements((prev) =>
        prev.filter((d) => d.question_id !== questionId)
      );
      // Invalidate session to refresh the grades table
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
    },
  });

  useEffect(() => {
    if (disagreementData) {
      setDisagreements(disagreementData);
    }
  }, [disagreementData]);

  // Auto-navigation on completion - switch to disagreements tab if there are any
  useEffect(() => {
    if (session?.status === "complete" && isGrading) {
      // Delay slightly to show final status
      const timeout = setTimeout(() => {
        // Check if there are unresolved disagreements
        const hasUnresolvedDisagreements = disagreementData && disagreementData.some((d: Disagreement) => !d.resolved);
        if (hasUnresolvedDisagreements) {
          setActiveTab("disagreements");
        } else {
          setActiveTab("review");
        }
        setIsGrading(false);
      }, 1500);
      return () => clearTimeout(timeout);
    }
  }, [session?.status, isGrading, sessionId, disagreementData]);

  // Auto-switch to review tab when all disagreements are resolved
  useEffect(() => {
    const unresolvedCount = disagreements.filter((d) => !d.resolved).length;
    if (activeTab === "disagreements" && unresolvedCount === 0 && disagreements.length > 0) {
      setActiveTab("review");
    }
  }, [disagreements, activeTab]);

  // Handle cancel button
  const handleCancel = async () => {
    if (confirm("Êtes-vous sûr ? Cela annulera toute la progression.")) {
      await cancelMutation.mutateAsync();
    }
  };

  // Get rotating waiting message (20 second interval)
  const waitingMessage = useRotatingMessage();

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

    // First filter by search query
    const filtered = session.graded_copies.filter((copy) => {
      if (!searchQuery) return true;
      const name = copy.student_name || copy.copy_id;
      return name.toLowerCase().includes(searchQuery.toLowerCase());
    });

    return filtered.sort((a, b) => {
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
        <main className="flex-1 container py-6">
          {/* Back button skeleton */}
          <Skeleton className="h-9 w-44 mb-4" />

          {/* Header skeleton */}
          <div className="mb-6">
            <div className="flex items-center gap-3 mb-3">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-6 w-20" />
            </div>
            <Skeleton className="h-5 w-64" />
          </div>

          {/* Table skeleton */}
          <div className="space-y-3">
            <Skeleton className="h-12 w-full" />
            {[...Array(8)].map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
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
            <h1 className="text-2xl font-bold mb-4">Correction non trouvée</h1>
            <Button asChild>
              <Link href="/dashboard">Retour au tableau de bord</Link>
            </Button>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  // Use state variable for isGrading to track during grading
  // Derived from session status but also managed during transitions
  // Also detect interrupted grading: status is "correction" but all copies are graded
  const wasInterrupted = session.status === "correction" &&
    session.copies_count > 0 &&
    session.graded_count === session.copies_count;
  const isGradingActive = session.status === "correction" && !wasInterrupted;
  const isComplete = session.status === "complete" || wasInterrupted;
  const hasDisagreements = disagreements.filter((d) => !d.resolved).length > 0;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-6">
        {/* Top row: Back button + Exam name + Status */}
        <div className="flex items-center justify-between mb-4 gap-2 flex-wrap">
          <div className="flex items-center gap-2 flex-wrap">
            <Button variant="ghost" asChild>
              <Link href="/dashboard">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Retour
              </Link>
            </Button>
            <span className="text-slate-300 hidden sm:inline">|</span>
            <EditableExamName
              sessionId={sessionId}
              subject={session.subject}
              fallback={`Session ${sessionId.slice(0, 8)}`}
            />
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            {progress.connected && isGradingActive && (
              <Badge variant="outline" className="text-green-600 border-green-600">
                Live
              </Badge>
            )}
            <SessionStatus status={wasInterrupted ? "complete" : session.status} />
            <span className="hidden sm:inline">·</span>
            <span className="hidden sm:inline">Créée le {formatDate(session.created_at)}</span>
          </div>
        </div>

        {/* Topic and Action button row */}
        {(session.topic || !isGradingActive) && (
          <div className="flex items-center justify-between mb-4 gap-4">
            {session.topic && (
              <span className="text-sm text-muted-foreground">{session.topic}</span>
            )}
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
                Démarrer la correction
              </Button>
            )}
          </div>
        )}

        {/* Progress Grid (during grading) */}
        {isGradingActive && (
          <Card className="mb-6 border-purple-200 dark:border-purple-900">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
                    Correction en cours
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
                  Annuler
                </Button>
              </div>
              {/* Progress summary */}
              <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2" role="status" aria-live="polite">
                <span>
                  {progress.completedCopies} / {progress.totalCopies || session.copies_count} copies
                  terminées
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

        {/* Tabs for Review/Analytics/Disagreements */}
        {isComplete && (
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <div className="flex items-center justify-between">
              <TabsList>
                <TabsTrigger value="review">Révision</TabsTrigger>
                <TabsTrigger value="analytics">Statistiques</TabsTrigger>
                {hasDisagreements && (
                  <TabsTrigger value="disagreements">
                    Désaccords
                    <Badge variant="warning" className="ml-2">
                      {disagreements.filter((d) => !d.resolved).length}
                    </Badge>
                  </TabsTrigger>
                )}
              </TabsList>
              <ExportButton sessionId={sessionId} />
            </div>

            {/* Review Tab */}
            <TabsContent value="review">
              {/* Search Bar - only show if more than 15 students */}
              {session.graded_copies && session.graded_copies.length > 15 && (
                <div className="mb-4 flex flex-col sm:flex-row sm:items-center gap-3">
                  <div className="relative flex-1 max-w-sm">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                    <Input
                      placeholder="Rechercher un élève..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-9 pr-9 h-10 bg-white border-slate-200 focus:border-purple-300 focus:ring-purple-200"
                    />
                    {searchQuery && (
                      <button
                        onClick={() => setSearchQuery("")}
                        className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 hover:text-slate-600 transition-colors"
                        aria-label="Effacer la recherche"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                  <span className="text-sm text-slate-500">
                    {sortedCopies.length} élève{sortedCopies.length !== 1 ? 's' : ''}
                    {searchQuery && (
                      <span className="text-slate-400"> sur {session.graded_copies.length}</span>
                    )}
                  </span>
                </div>
              )}

              {/* Mobile Card View */}
              <div className="md:hidden space-y-3">
                {sortedCopies.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Aucun élève ne correspond à la recherche
                  </div>
                ) : (
                  sortedCopies.map((graded) => {
                    const gradingAudit = graded.grading_audit as any;
                    return (
                      <Card key={graded.copy_id} className="shadow-sm border-slate-200 overflow-hidden">
                        <CardContent className="p-4">
                          {/* Student name and total */}
                          <div className="flex justify-between items-center mb-3 pb-3 border-b border-slate-100">
                            <EditableStudentName
                              sessionId={sessionId}
                              copyId={graded.copy_id}
                              studentName={graded.student_name}
                            />
                            <span className="font-bold text-lg">
                              <span className="text-slate-700">{graded.total_score.toFixed(1)}</span>
                              <span className="text-slate-400 mx-0.5">/</span>
                              <span className="text-slate-500">{graded.max_score}</span>
                            </span>
                          </div>

                          {/* Questions grid */}
                          <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                            {questionIds.map((questionId) => {
                              const questionHasDisagreement = gradingAudit?.questions?.[questionId]?.resolution?.agreement === false;
                              const maxPoints = graded.max_points_by_question?.[questionId] ?? session.question_weights?.[questionId] ?? 0;
                              const originalLLMGrade = gradingAudit?.questions?.[questionId]?.resolution?.final_grade;
                              const displayName = session.question_names?.[questionId] || questionId;

                              return (
                                <div key={questionId} className="flex justify-between items-center py-1.5">
                                  <span className="text-sm text-slate-600 truncate max-w-[45%]" title={displayName}>
                                    {displayName}
                                  </span>
                                  <EditableGradeCell
                                    sessionId={sessionId}
                                    copyId={graded.copy_id}
                                    questionId={questionId}
                                    grade={graded.grades[questionId] || 0}
                                    maxPoints={maxPoints}
                                    hasDisagreement={questionHasDisagreement}
                                    originalLLMGrade={originalLLMGrade}
                                  />
                                </div>
                              );
                            })}
                          </div>

                          {/* PDF Link */}
                          <div className="mt-3 pt-3 border-t border-slate-100">
                            <Button variant="ghost" size="sm" asChild className="w-full justify-center hover:bg-purple-50 hover:text-purple-700">
                              <a
                                href={`/api/sessions/${sessionId}/copies/${graded.copy_id}/pdf`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2"
                              >
                                <ExternalLink className="h-4 w-4" />
                                Voir le PDF
                              </a>
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })
                )}
              </div>

              {/* Desktop Table View */}
              <Card className="hidden md:block shadow-sm border border-slate-200 rounded-xl overflow-hidden">
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="hover:bg-transparent bg-slate-50/80">
                          <TableHead
                            className="cursor-pointer hover:bg-slate-100 sticky left-0 bg-slate-50 z-10 min-w-[160px] text-xs font-semibold text-slate-600 uppercase tracking-wider px-5 py-4 border-b-2 border-slate-200"
                            onClick={() => handleSort("copy_id")}
                          >
                            <div className="flex items-center gap-1">
                              Élève
                              {sortColumn === "copy_id" && (
                                sortDirection === "asc" ? (
                                  <ChevronUp className="h-4 w-4 text-purple-600" />
                                ) : (
                                  <ChevronDown className="h-4 w-4 text-purple-600" />
                                )
                              )}
                            </div>
                          </TableHead>
                          <TableHead
                            className="cursor-pointer hover:bg-slate-100 text-center min-w-[100px] text-xs font-semibold text-slate-600 uppercase tracking-wider px-5 py-4 border-b-2 border-slate-200"
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
                          {questionIds.map((questionId) => {
                            const maxPoints = session.question_weights?.[questionId] ?? 0;
                            const displayName = session.question_names?.[questionId] || questionId;
                            return (
                            <TableHead
                              key={questionId}
                              className="text-center min-w-[120px] text-xs font-semibold text-slate-600 uppercase tracking-wider px-4 py-4 border-b-2 border-slate-200"
                            >
                              <div className="flex flex-col items-center gap-1">
                                <span onClick={() => handleSort(questionId)} className="cursor-pointer hover:text-purple-600 flex items-center gap-1">
                                  <EditableQuestionName
                                    sessionId={sessionId}
                                    questionId={questionId}
                                    displayName={displayName}
                                    onNameChange={() => queryClient.invalidateQueries({ queryKey: ["session", sessionId] })}
                                  />
                                  {sortColumn === questionId && (
                                    sortDirection === "asc" ? (
                                      <ChevronUp className="h-3 w-3 text-purple-600" />
                                    ) : (
                                      <ChevronDown className="h-3 w-3 text-purple-600" />
                                    )
                                  )}
                                </span>
                                <span className="text-slate-400 font-normal normal-case text-[10px]">
                                  <EditableQuestionWeight
                                    sessionId={sessionId}
                                    questionId={questionId}
                                    weight={maxPoints}
                                    onWeightChange={() => queryClient.invalidateQueries({ queryKey: ["session", sessionId] })}
                                  />
                                </span>
                              </div>
                            </TableHead>
                            );
                          })}
                          <TableHead className="text-center min-w-[100px] text-xs font-semibold text-slate-600 uppercase tracking-wider px-5 py-4 border-b-2 border-slate-200">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedCopies.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={questionIds.length + 3} className="text-center py-8 text-muted-foreground">
                              Aucun élève ne correspond à la recherche
                            </TableCell>
                          </TableRow>
                        ) : (
                          sortedCopies.map((graded, index) => {
                            // Get grading audit to check for disagreements
                            const gradingAudit = graded.grading_audit as any;
                            const isEvenRow = index % 2 === 0;

                            return (
                              <TableRow
                                key={graded.copy_id}
                                className={`group transition-colors ${isEvenRow ? 'bg-white' : 'bg-slate-50/30'} hover:bg-purple-50/50`}
                              >
                                <TableCell className={`font-medium sticky left-0 z-10 px-5 py-4 ${isEvenRow ? 'bg-white' : 'bg-slate-50/30'} group-hover:bg-purple-50/50 transition-colors`}>
                                  <EditableStudentName
                                    sessionId={sessionId}
                                    copyId={graded.copy_id}
                                    studentName={graded.student_name}
                                  />
                                </TableCell>
                                <TableCell className="text-center font-semibold px-5 py-4">
                                  <span className="inline-flex items-center px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700">
                                    {graded.total_score.toFixed(1)}<span className="text-slate-400 mx-0.5">/</span>{graded.max_score}
                                  </span>
                                </TableCell>
                                {questionIds.map((questionId) => {
                                  // Check if this specific question has a disagreement
                                  const questionHasDisagreement = gradingAudit?.questions?.[questionId]?.resolution?.agreement === false;
                                  const maxPoints = graded.max_points_by_question?.[questionId] ?? session.question_weights?.[questionId] ?? 0;
                                  // Get the original LLM grade for reset functionality
                                  const originalLLMGrade = gradingAudit?.questions?.[questionId]?.resolution?.final_grade;

                                  return (
                                    <TableCell key={questionId} className="text-center px-4 py-4">
                                      <EditableGradeCell
                                        sessionId={sessionId}
                                        copyId={graded.copy_id}
                                        questionId={questionId}
                                        grade={graded.grades[questionId] || 0}
                                        maxPoints={maxPoints}
                                        hasDisagreement={questionHasDisagreement}
                                        originalLLMGrade={originalLLMGrade}
                                      />
                                    </TableCell>
                                  );
                                })}
                                <TableCell className="text-center px-5 py-4">
                                  <Button variant="ghost" size="sm" asChild className="hover:bg-purple-100 hover:text-purple-700">
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
                          })
                        )}
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
                      <CardTitle>Statistiques</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <dl className="grid grid-cols-2 gap-4">
                        <div>
                          <dt className="text-sm text-muted-foreground">Moyenne</dt>
                          <dd className="text-2xl font-bold">
                            {analytics.mean_score.toFixed(2)}
                          </dd>
                        </div>
                        <div>
                          <dt className="text-sm text-muted-foreground">Médiane</dt>
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
                            Écart-type
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
                    title="Distribution des notes"
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
                            copyId: disagreement.copy_id,
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
