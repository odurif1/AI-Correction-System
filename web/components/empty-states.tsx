"use client";

import { Button } from "@/components/ui/button";
import {
  FileText,
  Search,
  AlertTriangle,
  Inbox,
  Plus,
} from "lucide-react";
import Link from "next/link";
import { useI18n } from "@/lib/i18n";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: {
    label: string;
    href?: string;
    onClick?: () => void;
  };
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: EmptyStateProps) {
  return (
    <div className="text-center py-12">
      <div className="flex justify-center mb-4">
        {icon || <Inbox className="h-16 w-16 text-muted-foreground/50" />}
      </div>
      <h3 className="font-semibold text-lg mb-2">{title}</h3>
      <p className="text-muted-foreground mb-4 max-w-sm mx-auto">{description}</p>
      {action && (
        <Button asChild={!!action.href} onClick={action.onClick}>
          {action.href ? (
            <Link href={action.href}>
              <Plus className="h-4 w-4 mr-2" />
              {action.label}
            </Link>
          ) : (
            <>
              <Plus className="h-4 w-4 mr-2" />
              {action.label}
            </>
          )}
        </Button>
      )}
    </div>
  );
}

export function NoSessionsState() {
  return (
    <div className="text-center py-16">
      <div className="flex justify-center mb-4">
        <FileText className="h-16 w-16 text-muted-foreground/50" />
      </div>
      <h3 className="font-semibold text-xl mb-2">Aucune correction</h3>
      <p className="text-muted-foreground mb-6 max-w-sm mx-auto">
        Commencez par créer une nouvelle correction
      </p>
      <Button asChild>
        <Link href="/sessions/new">
          <Plus className="h-4 w-4 mr-2" />
          Nouvelle correction
        </Link>
      </Button>
    </div>
  );
}

export function NoResultsState({
  hasFilters,
  onClear,
}: {
  hasFilters: boolean;
  onClear?: () => void;
}) {
  const { t } = useI18n();

  return (
    <EmptyState
      icon={<Search className="h-16 w-16 text-muted-foreground/50" />}
      title={t("empty.noResults")}
      description={t("dashboard.noResultsDesc")}
      action={hasFilters && onClear ? { label: t("empty.clearFilters"), onClick: onClear } : undefined}
    />
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  const { t } = useI18n();

  return (
    <EmptyState
      icon={<AlertTriangle className="h-16 w-16 text-destructive/50" />}
      title={t("empty.error")}
      description={message}
      action={onRetry ? { label: t("empty.tryAgain"), onClick: onRetry } : undefined}
    />
  );
}

export function NoCopiesState() {
  const { t } = useI18n();

  return (
    <EmptyState
      icon={<FileText className="h-16 w-16 text-muted-foreground/50" />}
      title={t("empty.noCopies")}
      description={t("empty.noCopiesDesc")}
    />
  );
}

export function NoDisagreementsState() {
  const { t } = useI18n();

  return (
    <EmptyState
      icon={<Search className="h-16 w-16 text-success/50" />}
      title={t("sessionDetail.noDisagreements")}
      description={t("sessionDetail.noDisagreementsDesc")}
    />
  );
}
