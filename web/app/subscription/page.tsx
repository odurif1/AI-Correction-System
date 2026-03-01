"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth-context";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UsageBar } from "@/components/subscription/usage-bar";
import { BillingHistory } from "@/components/subscription/billing-history";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { CheckCircle2, Zap, Building, GraduationCap } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { ApiError } from "@/lib/api";
import { toast } from "sonner";

const plans = [
  {
    name: "Essentiel",
    price: "9€",
    period: "/mois",
    description: "Pour démarrer",
    icon: GraduationCap,
    tokens: "1.2M tokens",
    pages: "~120 pages",
    features: [
      "1.2M tokens (~120 pages)",
      "Double IA",
      "Export CSV et JSON",
      "Support par email",
    ],
    tier: "essentiel",
  },
  {
    name: "Pro",
    price: "27€",
    period: "/mois",
    description: "Pour les professeurs réguliers",
    icon: Zap,
    tokens: "6M tokens",
    pages: "~600 pages",
    features: [
      "6M tokens (~600 pages)",
      "Double IA",
      "Analytics avancés",
      "Export PDF annotés",
      "Support prioritaire",
    ],
    tier: "pro",
    popular: true,
  },
  {
    name: "Max",
    price: "72€",
    period: "/mois",
    description: "Pour faire profiter les collègues",
    icon: Building,
    tokens: "24M tokens",
    pages: "~2400 pages",
    features: [
      "24M tokens (~2400 pages)",
      "Double IA",
      "Analytics avancés",
      "Export PDF annotés",
      "Support téléphonique",
      "Formation incluse",
    ],
    tier: "max",
  },
];

export default function SubscriptionPage() {
  const { user } = useAuth();

  const currentTier = user?.subscription_tier || "free";

  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    plan: string;
    isUpgrade: boolean;
  }>({ open: false, plan: "", isUpgrade: false });

  const handleManageBilling = async () => {
    try {
      const response = await api.createPortalSession();
      window.location.href = response.portal_url;
    } catch (error) {
      if (error instanceof ApiError && error.status === 400) {
        toast.error("Aucun compte de facturation trouvé");
      } else {
        toast.error("Impossible d'ouvrir le portail de facturation. Réessayez.");
      }
    }
  };

  const handlePlanChange = (tier: string, isUpgrade: boolean) => {
    setConfirmDialog({ open: true, plan: tier, isUpgrade });
  };

  const onConfirm = async () => {
    try {
      await api.updateSubscription(confirmDialog.plan, !confirmDialog.isUpgrade);
      toast.success(
        confirmDialog.isUpgrade
          ? "Abonnement mis à niveau avec succès"
          : "Abonnement changé au prochain cycle de facturation"
      );
      // Refresh page to update subscription status
      window.location.reload();
    } catch (error) {
      toast.error("Impossible de changer d'abonnement. Réessayez.");
    } finally {
      setConfirmDialog({ ...confirmDialog, open: false });
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 container py-8">
        {/* Current plan */}
        <div className="max-w-4xl mx-auto mb-12">
          <h1 className="text-3xl font-bold mb-2">Abonnement</h1>
          <p className="text-muted-foreground mb-8">
            Gérez votre abonnement et vos tokens
          </p>

          <div className="mb-8">
            <UsageBar />
          </div>

          {currentTier !== 'free' && (
            <div className="flex justify-center mb-8">
              <Button variant="outline" onClick={handleManageBilling}>
                Gérer la facturation
              </Button>
            </div>
          )}
        </div>

        {/* Available plans */}
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-bold mb-6 text-center">Changer de plan</h2>

          <div className="grid md:grid-cols-3 gap-6">
            {plans.map((plan) => {
              const isCurrentPlan = currentTier === plan.tier;

              return (
                <Card
                  key={plan.name}
                  className={`relative flex flex-col ${
                    plan.popular
                      ? "border-purple-500 shadow-lg"
                      : ""
                  }`}
                >
                  {plan.popular && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <span className="bg-purple-600 text-white text-xs font-medium px-3 py-1 rounded-full">
                        Populaire
                      </span>
                    </div>
                  )}

                  <CardHeader className="text-center">
                    <div className="flex justify-center mb-2">
                      <plan.icon className={`h-8 w-8 ${plan.popular ? "text-purple-500" : "text-muted-foreground"}`} />
                    </div>
                    <CardTitle>{plan.name}</CardTitle>
                    <CardDescription>{plan.description}</CardDescription>
                  </CardHeader>

                  <CardContent className="flex-1">
                    <div className="text-center mb-4">
                      <span className="text-3xl font-bold">{plan.price}</span>
                      <span className="text-muted-foreground">{plan.period}</span>
                    </div>

                    <ul className="space-y-2 mb-6">
                      {plan.features.map((feature, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm">
                          <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>

                    <Button
                      className="w-full"
                      variant={isCurrentPlan ? "outline" : plan.popular ? "default" : "outline"}
                      disabled={isCurrentPlan}
                      onClick={() => {
                        if (!isCurrentPlan) {
                          const tierOrder = { free: 0, essentiel: 1, pro: 2, max: 3 };
                          const isUpgrade = tierOrder[plan.tier as keyof typeof tierOrder] > tierOrder[currentTier as keyof typeof tierOrder];
                          handlePlanChange(plan.tier, isUpgrade);
                        }
                      }}
                    >
                      {isCurrentPlan ? "Plan actuel" : `Passer à ${plan.name}`}
                    </Button>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Billing history */}
        <BillingHistory currentTier={currentTier} />
      </main>

      <Footer />

      {/* Plan change confirmation dialog */}
      <AlertDialog open={confirmDialog.open} onOpenChange={(open) => setConfirmDialog({ ...confirmDialog, open })}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmDialog.isUpgrade
                ? `Passer à ${confirmDialog.plan.charAt(0).toUpperCase() + confirmDialog.plan.slice(1)}`
                : `Revenir à ${confirmDialog.plan.charAt(0).toUpperCase() + confirmDialog.plan.slice(1)}`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmDialog.isUpgrade
                ? "Vous serez débité de la différence au prorata et aurez un accès immédiat."
                : "Votre abonnement changera au prochain cycle de facturation. Vous garderez votre accès actuel jusqu'à cette date."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Annuler</AlertDialogCancel>
            <AlertDialogAction onClick={onConfirm}>
              Confirmer
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
