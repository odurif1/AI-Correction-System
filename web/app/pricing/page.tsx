"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { CheckCircle2, Zap, Building, GraduationCap, Heart } from "lucide-react";

const plans = [
  {
    name: "Essentiel",
    price: "9€",
    period: "/mois",
    description: "Pour démarrer",
    icon: GraduationCap,
    features: [
      "1.2M tokens (~120 pages)",
      "Double IA",
      "Export CSV et JSON",
      "Support par email",
    ],
    cta: "Choisir",
    href: "/auth/register?plan=essentiel",
    popular: false,
  },
  {
    name: "Pro",
    price: "27€",
    period: "/mois",
    description: "Pour corriger souvent",
    icon: Zap,
    features: [
      "6M tokens (~600 pages)",
      "Double IA",
      "Analytics avancés",
      "Export PDF annotés",
      "Support prioritaire",
    ],
    cta: "Choisir",
    href: "/auth/register?plan=pro",
    popular: true,
  },
  {
    name: "Max",
    price: "72€",
    period: "/mois",
    description: "Pour faire profiter les collègues",
    icon: Building,
    features: [
      "24M tokens (~2400 pages)",
      "Double IA",
      "Analytics avancés",
      "Export PDF annotés",
      "Support téléphonique",
      "Formation incluse",
    ],
    cta: "Nous contacter",
    href: "mailto:contact@lacorrigeuse.fr?subject=Demande Max",
    popular: false,
  },
];

export default function PricingPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1">
        {/* Pricing Cards */}
        <section className="container py-12 md:py-16">
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto items-start">
            {plans.map((plan) => (
              <Card
                key={plan.name}
                className={`relative flex flex-col ${
                  plan.popular
                    ? "border-primary shadow-xl bg-gradient-to-br from-white to-purple-50 dark:from-background dark:to-purple-950/20"
                    : "bg-gradient-to-br from-white to-muted/50 dark:from-background dark:to-muted/20"
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="bg-gradient-to-r from-purple-600 to-blue-600 text-white text-sm font-medium px-4 py-1 rounded-full shadow-lg">
                      Le plus populaire
                    </span>
                  </div>
                )}

                <CardHeader className="text-center pt-8">
                  <div className="flex justify-center mb-4">
                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${
                      plan.popular
                        ? "bg-gradient-to-br from-purple-500 to-blue-500"
                        : "bg-muted"
                    }`}>
                      <plan.icon className={`h-7 w-7 ${
                        plan.popular ? "text-white" : "text-muted-foreground"
                      }`} />
                    </div>
                  </div>
                  <CardTitle className="text-2xl">{plan.name}</CardTitle>
                  <CardDescription>{plan.description}</CardDescription>
                </CardHeader>

                <CardContent className="flex-1">
                  <div className="text-center mb-6">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    <span className="text-muted-foreground">{plan.period}</span>
                  </div>

                  <ul className="space-y-3">
                    {plan.features.map((feature, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                        <span className="text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>

                <CardFooter className="pt-4">
                  <Button
                    className={`w-full ${plan.popular ? "bg-purple-600 hover:bg-purple-700" : ""}`}
                    variant={plan.popular ? "default" : "outline"}
                    size="lg"
                    asChild
                  >
                    <Link href={plan.href}>{plan.cta}</Link>
                  </Button>
                </CardFooter>
              </Card>
            ))}
          </div>
        </section>

        {/* Open Source Alternative */}
        <section className="bg-muted/30">
          <div className="container py-16">
            <div className="max-w-2xl mx-auto text-center space-y-4">
              <Heart className="h-10 w-10 text-red-500 fill-red-500 mx-auto" />
              <h2 className="text-2xl font-bold">
                Vous préférez l'auto-hébergement ?
              </h2>
              <p className="text-muted-foreground">
                La Corrigeuse est 100% open source. Hébergez-la sur votre propre serveur,
                avec vos propres clés API (coût des appels à votre charge).
              </p>
              <Button variant="outline" asChild>
                <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                  Voir la documentation
                </a>
              </Button>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-600 to-blue-600" />
          <div className="relative container py-16">
            <div className="text-center text-white space-y-4">
              <h2 className="text-2xl md:text-3xl font-bold">
                Prêt à récupérer vos week-end ?
              </h2>
              <p className="text-white/90">
                1 page offerte à l'inscription pour tester.
              </p>
              <Button size="lg" className="bg-white text-purple-600 hover:bg-purple-50" asChild>
                <Link href="/auth/register">
                  Créer mon compte
                </Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
