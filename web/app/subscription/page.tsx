"use client";

import { useAuth } from "@/lib/auth-context";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UsageBar } from "@/components/subscription/usage-bar";
import { CheckCircle2, Zap, Building, GraduationCap } from "lucide-react";
import Link from "next/link";

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
                    >
                      {isCurrentPlan ? "Plan actuel" : "Choisir"}
                    </Button>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
