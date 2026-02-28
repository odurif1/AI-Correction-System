"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

interface UsageBarProps {
  className?: string;
}

interface SubscriptionStatus {
  tier: string;
  tokens_used: number;
  monthly_limit: number;
  remaining_tokens: number;
  has_monthly_reset: boolean;
}

export function UsageBar({ className = "" }: UsageBarProps) {
  const [status, setStatus] = useState<SubscriptionStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getSubscriptionStatus()
      .then(setStatus)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading || !status) {
    return <div className={`h-20 bg-gray-100 rounded animate-pulse ${className}`} />;
  }

  const percentage = Math.min((status.tokens_used / status.monthly_limit) * 100, 100);
  const remaining = Math.max(status.monthly_limit - status.tokens_used, 0);

  // Color based on percentage
  const barColor =
    percentage > 90 ? "bg-red-500" :
    percentage > 70 ? "bg-yellow-500" :
    "bg-purple-600";

  return (
    <div className={`bg-white rounded-lg shadow p-4 ${className}`}>
      <div className="flex justify-between items-center mb-2">
        <div>
          <p className="text-sm text-gray-600">
            {status.tier === 'free' ? 'One-shot (Demo)' : 'Ce mois'}
          </p>
          <p className="text-xs text-gray-500 capitalize">
            Tier {status.tier}
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium">
            {status.tokens_used.toLocaleString()} / {status.monthly_limit.toLocaleString()} tokens
          </p>
          <p className="text-xs text-gray-500">
            {remaining.toLocaleString()} restants
          </p>
        </div>
      </div>

      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all ${barColor}`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {percentage > 90 && (
        <p className="text-xs text-red-600 mt-2">
          Limite presque atteinte - pensez à upgrader
        </p>
      )}
    </div>
  );
}
