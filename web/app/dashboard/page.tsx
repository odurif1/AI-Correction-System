"use client";

import { useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { DashboardSkeleton } from "@/components/loading-skeletons";
import { ConfirmDialog } from "@/components/confirmation-dialog";
import { SessionCardsGrid } from "@/components/session-cards";
import { NoSessionsState, ErrorState } from "@/components/empty-states";
import { api } from "@/lib/api";
import { Plus } from "lucide-react";
import type { Session } from "@/lib/types";
import { useInView } from "react-intersection-observer";
import { useEffect } from "react";

const SESSIONS_PER_PAGE = 20;

export default function DashboardPage() {
  const queryClient = useQueryClient();

  // State
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  // Infinite scroll query
  const {
    data,
    isLoading,
    error,
    refetch,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ["sessions"],
    queryFn: async ({ pageParam = 0 }) => {
      const result = await api.listSessions();
      // Client-side pagination since backend doesn't support offset/limit yet
      const start = pageParam * SESSIONS_PER_PAGE;
      const end = start + SESSIONS_PER_PAGE;
      const paginatedSessions = result.sessions.slice(start, end);
      return {
        sessions: paginatedSessions,
        total: result.sessions.length,
        page: pageParam,
        hasMore: end < result.sessions.length,
      };
    },
    initialPageParam: 0,
    getNextPageParam: (lastPage) => {
      return lastPage.hasMore ? lastPage.page + 1 : undefined;
    },
  });

  // Flatten sessions from all pages
  const allSessions = useMemo(() => {
    return data?.pages.flatMap((page) => page.sessions) || [];
  }, [data]);

  // Filter sessions by status
  const filteredSessions = useMemo(() => {
    if (statusFilter === "all") {
      return allSessions;
    }
    return allSessions.filter((session: Session) => session.status === statusFilter);
  }, [allSessions, statusFilter]);

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (sessionId: string) => api.deleteSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      setDeleteDialogOpen(false);
      setSessionToDelete(null);
    },
    onError: (error: Error) => {
      console.error("Delete error:", error.message);
    },
  });

  // Infinite scroll trigger
  const { ref: loadMoreRef, inView } = useInView({
    threshold: 0,
    rootMargin: "100px",
  });

  // Load more when scroll trigger is in view
  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const handleDeleteClick = (sessionId: string) => {
    setSessionToDelete(sessionId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (sessionToDelete) {
      deleteMutation.mutate(sessionToDelete);
    }
  };

  if (error) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container py-8">
          <ErrorState message="Impossible de charger les corrections" onRetry={() => refetch()} />
        </main>
        <Footer />
      </div>
    );
  }

  // Get total count from first page
  const totalCount = data?.pages[0]?.total || 0;
  const hasSessions = totalCount > 0;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {isLoading && !data ? (
          <DashboardSkeleton />
        ) : !hasSessions ? (
          <NoSessionsState />
        ) : (
          <>
            {/* Header with CTA and filter */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">Corrections</h1>
                <p className="text-muted-foreground mt-1">
                  {filteredSessions.length} correction{filteredSessions.length !== 1 ? "s" : ""}
                </p>
              </div>

              <div className="flex items-center gap-3">
                {/* Status filter */}
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="flex h-10 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value="all">Tous les statuts</option>
                  <option value="diagnostic">Diagnostic</option>
                  <option value="correction">Correction en cours</option>
                  <option value="complete">Terminée</option>
                  <option value="error">Erreur</option>
                </select>

                {/* New session CTA */}
                <Button
                  asChild
                  className="bg-purple-600 hover:bg-purple-700 text-white"
                >
                  <Link href="/sessions/new">
                    <Plus className="h-4 w-4 mr-2" />
                    Nouvelle correction
                  </Link>
                </Button>
              </div>
            </div>

            {/* Sessions grid */}
            {filteredSessions.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-muted-foreground">Aucune correction ne correspond au filtre.</p>
              </div>
            ) : (
              <>
                <SessionCardsGrid
                  sessions={filteredSessions}
                  onDelete={handleDeleteClick}
                />

                {/* Load more trigger */}
                {hasNextPage && (
                  <div ref={loadMoreRef} className="flex justify-center py-8">
                    {isFetchingNextPage && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-600" />
                        <span>Chargement...</span>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </>
        )}
      </main>

      <Footer />

      {/* Delete Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        onOpenChange={setDeleteDialogOpen}
        title="Supprimer la correction"
        description="Êtes-vous sûr de vouloir supprimer cette correction ? Cette action est irréversible."
        confirmLabel="Supprimer"
        onConfirm={handleDeleteConfirm}
        variant="destructive"
        loading={deleteMutation.isPending}
      />
    </div>
  );
}
